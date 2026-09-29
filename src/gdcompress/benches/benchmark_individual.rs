use std::hint::black_box;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::prelude::*;
use gdcompress::{
    BaseSelectionContext, BitDataSet, CompressedData, DecompressRandomAccessHandle, EntroGdError,
    EntropyScoredContext, PreEncodeContext,
};

// ── CSV collection ────────────────────────────────────────────────────────────

struct BenchRecord {
    stage: &'static str,
    impl_name: String,
    file: String,
    sample_time_ns: u128,
    source_bytes: u64,
    color_model_seed: String,
    pixel_grouping_seed: String,
    group_transform_seed: String,
    grouped_pixels_seed: String,
}

static BENCH_RECORDS: LazyLock<Mutex<Vec<BenchRecord>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ── Harness config ────────────────────────────────────────────────────────────

const N_RUNS: usize = 3;

// ── Canonical pipeline config ─────────────────────────────────────────────────

const NORMAL_PATIENCE: usize = 10;
const THRESHOLD_PATIENCE: usize = 5;
const ADAPTIVE_PATIENCE: usize = 5;
const ENTROPY_THRESHOLD: f64 = 0.75;
const WEIGHT_DECAY: f64 = 0.5;
const BASE_BIT_IMPL: BaseBitImpl = BaseBitImpl::Naive;
const IMAGE_DIR: &str = "data/bench_datasets/kodak_dataset";

// ── Seed preprocessing config ─────────────────────────────────────────────────
// Edit SEED_PREPROCESSING_CONFIGS to change which preprocessing configs are used
// as seeds for all downstream benchmark stages.

#[derive(Clone, Copy)]
struct SeedPreprocessing {
    color_model: ImageColorModel,
    group_w: u32,
    group_h: u32,
    grouping_transform: ImageGroupingTransform,
}

impl SeedPreprocessing {
    fn color_model_label(self) -> &'static str {
        match self.color_model {
            ImageColorModel::Rgb => "rgb",
            ImageColorModel::YCoCg => "ycocg",
            ImageColorModel::YCoCgR => "ycocgr",
        }
    }

    fn pixel_grouping_label(self) -> String {
        format!("{}x{}", self.group_w, self.group_h)
    }

    fn group_transform_label(self) -> &'static str {
        match self.grouping_transform {
            ImageGroupingTransform::Raw => "raw",
            ImageGroupingTransform::ForFirstPixel => "for_first_pixel",
            ImageGroupingTransform::ForMin => "for_min",
        }
    }

    fn grouped_pixels(self) -> u32 {
        self.group_w * self.group_h
    }

    fn artifact_label(self) -> String {
        format!(
            "{}_{}x{}_{}",
            self.color_model_label(),
            self.group_w,
            self.group_h,
            self.group_transform_label()
        )
    }

    fn process(self, path: PathBuf) -> Result<BitDataSet, EntroGdError> {
        let image = OpenImage.process(path)?;
        BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: self.color_model,
            pixel_grouping: PixelGrouping::new(self.group_w, self.group_h),
            grouping_transform: self.grouping_transform,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
        .process(image)
    }
}

#[derive(Clone, Copy)]
enum EntropyImpl {
    Naive,
    Batched,
}

impl EntropyImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Batched => "batched",
        }
    }

    fn process(self, input: BitDataSet) -> Result<EntropyScoredContext, EntroGdError> {
        match self {
            Self::Naive => Entropy {}.process(input),
            Self::Batched => Entropy {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum SelectBasesImpl {
    Naive {
        patience: usize,
    },
    Threshold {
        base_bit_impl: BaseBitImpl,
        patience: usize,
    },
    Adaptive {
        base_bit_impl: BaseBitImpl,
        patience: usize,
    },
}

impl SelectBasesImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Naive { .. } => "naive",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::Naive,
                ..
            } => "threshold_naive",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
                ..
            } => "threshold_hyper_log_log_count",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::Naive,
                ..
            } => "adaptive_naive",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
                ..
            } => "adaptive_hyper_log_log_count",
        }
    }

    fn process(self, input: EntropyScoredContext) -> Result<BaseSelectionContext, EntroGdError> {
        match self {
            Self::Naive { patience } => SelectBases { patience }.process(input),
            Self::Threshold {
                base_bit_impl,
                patience,
            } => SelectBasesThreshold {
                patience,
                base_bit_impl,
                entropy_threshold: ENTROPY_THRESHOLD,
            }
            .process(input),
            Self::Adaptive {
                base_bit_impl,
                patience,
            } => SelectBasesAdaptive {
                width_decay: WEIGHT_DECAY,
                patience,
                base_bit_impl,
            }
            .process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum BuildBaseTableImpl {
    Unsorted { base_bit_impl: BaseBitImpl },
    Sorted { base_bit_impl: BaseBitImpl },
}

impl BuildBaseTableImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Unsorted {
                base_bit_impl: BaseBitImpl::Naive,
            } => "build_base_table_naive",
            Self::Sorted {
                base_bit_impl: BaseBitImpl::Naive,
            } => "build_sorted_base_table_naive",
            _ => unreachable!(),
        }
    }

    fn select_bases(self, entropy_ctx: EntropyScoredContext) -> BaseSelectionContext {
        let base_bit_impl = match self {
            Self::Unsorted { base_bit_impl } | Self::Sorted { base_bit_impl } => base_bit_impl,
        };
        SelectBasesThreshold {
            patience: THRESHOLD_PATIENCE,
            base_bit_impl,
            entropy_threshold: ENTROPY_THRESHOLD,
        }
        .process(entropy_ctx)
        .unwrap()
    }

    fn process(self, input: BaseSelectionContext) -> Result<PreEncodeContext, EntroGdError> {
        match self {
            Self::Unsorted { .. } => BuildBaseTable {}.process(input),
            Self::Sorted { .. } => BuildSortedBaseTable {}.process(input),
        }
    }
}

struct EncodeSeed {
    pre_encode: PreEncodeContext,
}

#[derive(Clone, Copy)]
enum EncodeImpl {
    Normal,
    RleOffset,
    Huffman,
}

impl EncodeImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, input: EncodeSeed) -> Result<CompressedData, EntroGdError> {
        match self {
            Self::Normal => EncodeData {}.process(input.pre_encode),
            Self::RleOffset => EncodeDataOffsetRLE {}.process(input.pre_encode),
            Self::Huffman => EncodeDataHuffman {}.process(input.pre_encode),
        }
    }
}

#[derive(Clone, Copy)]
enum DecodeDeltaImpl {
    Unary,
    Fixed,
}

impl DecodeDeltaImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Unary => "unary",
            Self::Fixed => "fixed",
        }
    }
}

#[derive(Clone, Copy)]
enum DeltaEncodeImpl {
    Unary,
    Fixed,
}

impl DeltaEncodeImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Unary => "unary",
            Self::Fixed => "fixed",
        }
    }

    fn process(self, input: CompressedData) -> Result<CompressedData, EntroGdError> {
        match self {
            Self::Unary => DeltaEncodeBaseTable {}.process(input),
            Self::Fixed => DeltaEncodeBaseTableFixed {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum SaveIgdImpl {
    Normal,
    RleOffset,
    Huffman,
}

impl SaveIgdImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, compressed: CompressedData, path: PathBuf) -> Result<PathBuf, EntroGdError> {
        SaveIgdFile { output_path: path }.process(compressed)
    }
}

#[derive(Clone, Copy)]
enum LoadIgdImpl {
    Normal,
    RleOffset,
    Huffman,
}

impl LoadIgdImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, path: PathBuf) -> Result<CompressedData, EntroGdError> {
        LoadIgdFile {}.process(path)
    }
}

#[derive(Clone, Copy)]
enum DecompressFileImpl {
    Normal,
    RleOffset,
    Huffman,
}

impl DecompressFileImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, input: CompressedData) -> Result<BitDataSet, EntroGdError> {
        DecompressFileData {}.process(input)
    }
}

#[derive(Clone, Copy)]
enum DecompressRowsImpl {
    Normal,
    RleOffset,
    Huffman,
}

impl DecompressRowsImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(
        self,
        handle: Rc<DecompressRandomAccessHandle>,
        indices: Vec<usize>,
    ) -> Result<BitDataSet, EntroGdError> {
        (*handle).clone().decompress_samples(&indices)
    }
}

// ── Seed preprocessing configs ────────────────────────────────────────────────
// Add / remove / edit rows here to change which preprocessing configs seed the
// downstream benchmark stages (Entropy, SelectBases, Encode, Save, Load, Decompress).

macro_rules! seed_configs {
    ( $( ($cm:ident, $w:expr, $h:expr, $gt:ident) ),* $(,)? ) => {
        &[
            $(
                SeedPreprocessing {
                    color_model: ImageColorModel::$cm,
                    group_w: $w,
                    group_h: $h,
                    grouping_transform: ImageGroupingTransform::$gt,
                }
            ),*
        ]
    };
}

const SEED_PREPROCESSING_CONFIGS: &[SeedPreprocessing] = seed_configs![
    (YCoCgR, 1, 1, Raw),
    (YCoCgR, 2, 1, Raw),
    (YCoCgR, 2, 1, ForMin),
    (YCoCgR, 2, 1, ForFirstPixel),
    (YCoCgR, 3, 1, Raw),
    (YCoCgR, 3, 1, ForMin),
    (YCoCgR, 3, 1, ForFirstPixel),
    (YCoCgR, 2, 2, Raw),
    (YCoCgR, 2, 2, ForMin),
    (YCoCgR, 2, 2, ForFirstPixel),
    (YCoCgR, 2, 3, Raw),
    (YCoCgR, 2, 3, ForMin),
    (YCoCgR, 2, 3, ForFirstPixel),
    (YCoCgR, 3, 3, Raw),
    (YCoCgR, 3, 3, ForMin),
    (YCoCgR, 3, 3, ForFirstPixel),
];

// Rgb vs YCoCgR at 1x1/Raw to isolate color model conversion cost.
const BUILD_IMAGE_COLOR_MODEL_CONFIGS: &[SeedPreprocessing] =
    seed_configs![(Rgb, 1, 1, Raw), (YCoCgR, 1, 1, Raw),];

// YCoCgR with various groupings × all 3 transforms to isolate grouping cost.
const BUILD_IMAGE_GROUPING_CONFIGS: &[SeedPreprocessing] = seed_configs![
    (YCoCgR, 2, 1, Raw),
    (YCoCgR, 2, 1, ForFirstPixel),
    (YCoCgR, 2, 1, ForMin),
    (YCoCgR, 3, 1, Raw),
    (YCoCgR, 3, 1, ForFirstPixel),
    (YCoCgR, 3, 1, ForMin),
    (YCoCgR, 2, 2, Raw),
    (YCoCgR, 2, 2, ForFirstPixel),
    (YCoCgR, 2, 2, ForMin),
    (YCoCgR, 2, 3, Raw),
    (YCoCgR, 2, 3, ForFirstPixel),
    (YCoCgR, 2, 3, ForMin),
    (YCoCgR, 3, 3, Raw),
    (YCoCgR, 3, 3, ForFirstPixel),
    (YCoCgR, 3, 3, ForMin),
];

// ── Standard arrays ───────────────────────────────────────────────────────────

const ENTROPY_IMPLS: [EntropyImpl; 2] = [EntropyImpl::Naive, EntropyImpl::Batched];

const SELECT_BASES_IMPLS: [SelectBasesImpl; 5] = [
    SelectBasesImpl::Naive {
        patience: NORMAL_PATIENCE,
    },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::Naive,
        patience: THRESHOLD_PATIENCE,
    },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::HyperLogLogCount,
        patience: THRESHOLD_PATIENCE,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::Naive,
        patience: ADAPTIVE_PATIENCE,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::HyperLogLogCount,
        patience: ADAPTIVE_PATIENCE,
    },
];

const BUILD_BASE_TABLE_IMPLS: [BuildBaseTableImpl; 2] = [
    BuildBaseTableImpl::Unsorted {
        base_bit_impl: BaseBitImpl::Naive,
    },
    BuildBaseTableImpl::Sorted {
        base_bit_impl: BaseBitImpl::Naive,
    },
];

const ENCODE_IMPLS: [EncodeImpl; 3] = [
    EncodeImpl::Normal,
    EncodeImpl::RleOffset,
    EncodeImpl::Huffman,
];

const DELTA_ENCODE_IMPLS: [DeltaEncodeImpl; 2] = [DeltaEncodeImpl::Unary, DeltaEncodeImpl::Fixed];
const DECODE_DELTA_IMPLS: [DecodeDeltaImpl; 2] = [DecodeDeltaImpl::Unary, DecodeDeltaImpl::Fixed];
const SAVE_IGD_IMPLS: [SaveIgdImpl; 3] = [
    SaveIgdImpl::Normal,
    SaveIgdImpl::RleOffset,
    SaveIgdImpl::Huffman,
];
const LOAD_IGD_IMPLS: [LoadIgdImpl; 3] = [
    LoadIgdImpl::Normal,
    LoadIgdImpl::RleOffset,
    LoadIgdImpl::Huffman,
];
const DECOMPRESS_FILE_IMPLS: [DecompressFileImpl; 3] = [
    DecompressFileImpl::Normal,
    DecompressFileImpl::RleOffset,
    DecompressFileImpl::Huffman,
];
const DECOMPRESS_ROWS_IMPLS: [DecompressRowsImpl; 3] = [
    DecompressRowsImpl::Normal,
    DecompressRowsImpl::RleOffset,
    DecompressRowsImpl::Huffman,
];

// ── Case preparation ──────────────────────────────────────────────────────────

struct CompressedDataSeeds {
    normal: CompressedData,
    rle_offset: CompressedData,
    huffman: CompressedData,
}

struct IgdPaths {
    normal: PathBuf,
    rle_offset: PathBuf,
    huffman: PathBuf,
}

struct DecompressRowsSeeds {
    normal: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
    rle_offset: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
    huffman: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
}

struct PreparedCase {
    name: String,
    source_size: u64,
    preprocessing: SeedPreprocessing,
    bit_data_seed: BitDataSet,
    entropy_seed: EntropyScoredContext,
    // Heavy seeds are only built when BENCH_FILTER selects a stage that consumes them;
    // otherwise they stay None so a filtered run skips the work. The closures in
    // `benchmark_filter_steps` unwrap them, which is safe because a closure only runs
    // for a stage that was selected (and therefore prepared).
    base_table_ctx_seed: Option<PreEncodeContext>,
    pre_delta_compressed_seed: Option<CompressedData>,
    delta_unary_seed: Option<CompressedData>,
    delta_fixed_seed: Option<CompressedData>,
    compressed_seeds: Option<CompressedDataSeeds>,
    loaded_compressed_seeds: Option<CompressedDataSeeds>,
    igd_paths: IgdPaths,
    rows_context_seeds: Option<DecompressRowsSeeds>,
}

/// True when `stage` is selected by the current `BENCH_FILTER` (empty filter = all
/// stages). Mirrors the gate inside `bench_step_group` so `prepare_case` only builds
/// the seed artifacts the selected stages will actually use.
fn stage_selected(stage: &str) -> bool {
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    filter.is_empty() || stage.contains(filter.as_str())
}

fn canonical_select_bases(input: EntropyScoredContext) -> BaseSelectionContext {
    SelectBasesThreshold {
        patience: THRESHOLD_PATIENCE,
        base_bit_impl: BASE_BIT_IMPL,
        entropy_threshold: ENTROPY_THRESHOLD,
    }
    .process(input)
    .unwrap()
}

fn igd_artifact_path(name: &str, preprocessing_label: &str, encoding: &str) -> PathBuf {
    PathBuf::from(format!(
        "target/bench-artifacts/{}-{}-{}.igd",
        name, preprocessing_label, encoding
    ))
}

fn prepare_case(image_path: PathBuf, preprocessing: SeedPreprocessing) -> PreparedCase {
    let name = image_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    // Which downstream stages BENCH_FILTER selects. We only build the seed artifacts the
    // selected stages consume, so e.g. `BENCH_FILTER=Delta` skips the unsorted base table,
    // the extra encodes, and all igd save/load + random-access decode work.
    let want_encode_data = stage_selected("EncodeData");
    let want_save = stage_selected("SaveIgdFile");
    let want_load = stage_selected("LoadIgdFile");
    let want_decompress_file = stage_selected("DecompressFileData");
    let want_decompress_rows = stage_selected("DecompressRowsData");
    let want_delta_decode = stage_selected("DecodeDeltaBaseTable");
    let want_delta = stage_selected("DeltaEncodeBaseTable") || want_delta_decode;

    // The compressed seeds (and the encodes producing them) feed every igd/decompress stage.
    let want_compressed_seeds =
        want_save || want_load || want_decompress_file || want_decompress_rows;
    let want_igd_files = want_load || want_decompress_file || want_decompress_rows;
    let want_loaded = want_decompress_file || want_decompress_rows;
    let want_unsorted_ctx = want_encode_data || want_compressed_seeds;

    let bit_data_seed = preprocessing.process(image_path.clone()).unwrap();
    let source_size = (bit_data_seed.data.num_rows * bit_data_seed.data.chunk_size / 8) as u64;

    let entropy_seed = Entropy {}.process(bit_data_seed.clone()).unwrap();

    let base_table_ctx_seed = want_unsorted_ctx.then(|| {
        let sel = canonical_select_bases(entropy_seed.clone());
        BuildBaseTable {}.process(sel).unwrap()
    });

    // Delta coding only engages when the base table is sorted (it bails to Raw unless
    // `entropy_sorted_column_order` is Some, which only `BuildSortedBaseTable` sets). Seed
    // the delta stages from a sorted context so the codec actually runs — encoding the
    // unsorted `base_table_ctx_seed` would make every delta sample a no-op.
    let pre_delta_compressed_seed = want_delta.then(|| {
        let sel = canonical_select_bases(entropy_seed.clone());
        let sorted = BuildSortedBaseTable {}.process(sel).unwrap();
        EncodeData {}.process(sorted).unwrap()
    });
    let delta_unary_seed = want_delta_decode.then(|| {
        DeltaEncodeBaseTable {}
            .process(pre_delta_compressed_seed.clone().unwrap())
            .unwrap()
    });
    let delta_fixed_seed = want_delta_decode.then(|| {
        DeltaEncodeBaseTableFixed {}
            .process(pre_delta_compressed_seed.clone().unwrap())
            .unwrap()
    });

    let pre_label = preprocessing.artifact_label();
    let igd_paths = IgdPaths {
        normal: igd_artifact_path(&name, &pre_label, "normal"),
        rle_offset: igd_artifact_path(&name, &pre_label, "rle_offset"),
        huffman: igd_artifact_path(&name, &pre_label, "huffman"),
    };

    let compressed_seeds = want_compressed_seeds.then(|| {
        let base_ctx = base_table_ctx_seed
            .clone()
            .expect("unsorted base table context is built whenever compressed seeds are");
        let sorted_ctx = {
            let sel = canonical_select_bases(entropy_seed.clone());
            BuildSortedBaseTable {}.process(sel).unwrap()
        };
        let normal = EncodeData {}.process(base_ctx.clone()).unwrap();
        let rle_offset = EncodeDataOffsetRLE {}.process(base_ctx).unwrap();
        let huffman = EncodeDataHuffman {}.process(sorted_ctx).unwrap();

        if want_igd_files {
            SaveIgdFile {
                output_path: igd_paths.normal.clone(),
            }
            .process(normal.clone())
            .unwrap();
            SaveIgdFile {
                output_path: igd_paths.rle_offset.clone(),
            }
            .process(rle_offset.clone())
            .unwrap();
            SaveIgdFile {
                output_path: igd_paths.huffman.clone(),
            }
            .process(huffman.clone())
            .unwrap();
        }

        CompressedDataSeeds {
            normal,
            rle_offset,
            huffman,
        }
    });

    let loaded_compressed_seeds = want_loaded.then(|| CompressedDataSeeds {
        normal: LoadIgdFile {}.process(igd_paths.normal.clone()).unwrap(),
        rle_offset: LoadIgdFile {}
            .process(igd_paths.rle_offset.clone())
            .unwrap(),
        huffman: LoadIgdFile {}.process(igd_paths.huffman.clone()).unwrap(),
    });

    let rows_context_seeds = want_decompress_rows.then(|| {
        let loaded = loaded_compressed_seeds
            .as_ref()
            .expect("loaded seeds are built whenever random-access handles are");
        let rows: Vec<usize> = (0..bit_data_seed.data.num_rows).collect();
        DecompressRowsSeeds {
            normal: (
                Rc::new(DecompressRandomAccessHandle::new(loaded.normal.clone()).unwrap()),
                rows.clone(),
            ),
            rle_offset: (
                Rc::new(DecompressRandomAccessHandle::new(loaded.rle_offset.clone()).unwrap()),
                rows.clone(),
            ),
            huffman: (
                Rc::new(DecompressRandomAccessHandle::new(loaded.huffman.clone()).unwrap()),
                rows,
            ),
        }
    });

    PreparedCase {
        name,
        source_size,
        preprocessing,
        bit_data_seed,
        entropy_seed,
        base_table_ctx_seed,
        pre_delta_compressed_seed,
        delta_unary_seed,
        delta_fixed_seed,
        compressed_seeds,
        loaded_compressed_seeds,
        igd_paths,
        rows_context_seeds,
    }
}

struct ImageCase {
    name: String,
    path: PathBuf,
    source_size: u64,
}

fn sorted_image_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(IMAGE_DIR)
        .unwrap_or_else(|e| panic!("cannot read {IMAGE_DIR}: {e}"))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let ext = path.extension()?.to_string_lossy().to_lowercase();
            (ext == "png" || ext == "jpg" || ext == "jpeg").then_some(path)
        })
        .collect();
    paths.sort();
    paths
}

fn discover_image_cases() -> Vec<ImageCase> {
    sorted_image_paths()
        .into_iter()
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let source_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            ImageCase {
                name,
                path,
                source_size,
            }
        })
        .collect()
}

fn bench_build_image_color_model_step(image_cases: &[ImageCase]) {
    let stage_name = "BuildImageColorModel";
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage_name.contains(filter.as_str()) {
        return;
    }
    let total = image_cases.len() * BUILD_IMAGE_COLOR_MODEL_CONFIGS.len();
    let mut done = 0;
    for case in image_cases {
        for &pre in BUILD_IMAGE_COLOR_MODEL_CONFIGS {
            let label = pre.color_model_label();
            done += 1;
            eprintln!("[{done}/{total}] {stage_name}/{label}/{}", case.name);
            let _ = black_box(pre.process(case.path.clone()));
            for _ in 0..N_RUNS {
                let start = Instant::now();
                let output = pre.process(case.path.clone()).unwrap();
                let elapsed = start.elapsed();
                black_box(output);
                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    stage: stage_name,
                    impl_name: label.to_string(),
                    file: case.name.clone(),
                    sample_time_ns: elapsed.as_nanos(),
                    source_bytes: case.source_size,
                    color_model_seed: String::new(),
                    pixel_grouping_seed: String::new(),
                    group_transform_seed: String::new(),
                    grouped_pixels_seed: String::new(),
                });
            }
        }
    }
}

fn bench_build_image_grouping_step(image_cases: &[ImageCase]) {
    let stage_name = "BuildImageGrouping";
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage_name.contains(filter.as_str()) {
        return;
    }
    let total = image_cases.len() * BUILD_IMAGE_GROUPING_CONFIGS.len();
    let mut done = 0;
    for case in image_cases {
        for &pre in BUILD_IMAGE_GROUPING_CONFIGS {
            let label = format!(
                "{}_{}",
                pre.pixel_grouping_label(),
                pre.group_transform_label()
            );
            done += 1;
            eprintln!("[{done}/{total}] {stage_name}/{label}/{}", case.name);
            let _ = black_box(pre.process(case.path.clone()));
            for _ in 0..N_RUNS {
                let start = Instant::now();
                let output = pre.process(case.path.clone()).unwrap();
                let elapsed = start.elapsed();
                black_box(output);
                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    stage: stage_name,
                    impl_name: label.clone(),
                    file: case.name.clone(),
                    sample_time_ns: elapsed.as_nanos(),
                    source_bytes: case.source_size,
                    color_model_seed: String::new(),
                    pixel_grouping_seed: String::new(),
                    group_transform_seed: String::new(),
                    grouped_pixels_seed: String::new(),
                });
            }
        }
    }
}

// ── Generic benchmark harness ─────────────────────────────────────────────────

fn bench_step_group<ImplType, Input, Output, LabelFn, InputFn, RunFn, SeedMetaFn, SourceBytesFn>(
    stage_name: &'static str,
    prepared_cases: &[PreparedCase],
    implementations: &[ImplType],
    impl_label: LabelFn,
    make_input: InputFn,
    run_impl: RunFn,
    seed_meta: SeedMetaFn,
    source_bytes_fn: SourceBytesFn,
) where
    ImplType: Copy,
    LabelFn: Fn(ImplType) -> &'static str + Copy,
    InputFn: Fn(ImplType, &PreparedCase) -> Input + Copy,
    RunFn: Fn(ImplType, Input) -> Result<Output, EntroGdError> + Copy,
    SeedMetaFn: Fn(&PreparedCase) -> (String, String, String, String) + Copy,
    SourceBytesFn: Fn(&PreparedCase) -> u64 + Copy,
{
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage_name.contains(filter.as_str()) {
        return;
    }
    let total = prepared_cases.len() * implementations.len();
    let mut done = 0;
    for case in prepared_cases {
        for &implementation in implementations {
            let label = impl_label(implementation);
            done += 1;
            eprintln!("[{done}/{total}] {stage_name}/{label}/{}", case.name);

            // warmup: avoids measuring cold-start paging / branch-predictor effects
            let _ = black_box(run_impl(implementation, make_input(implementation, case)));

            for _ in 0..N_RUNS {
                let input = make_input(implementation, case);
                let start = Instant::now();
                let output = run_impl(implementation, input).unwrap();
                let elapsed = start.elapsed();
                black_box(output);
                let (cm, pg, gt, gp) = seed_meta(case);
                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    stage: stage_name,
                    impl_name: label.to_string(),
                    file: case.name.clone(),
                    sample_time_ns: elapsed.as_nanos(),
                    source_bytes: source_bytes_fn(case),
                    color_model_seed: cm,
                    pixel_grouping_seed: pg,
                    group_transform_seed: gt,
                    grouped_pixels_seed: gp,
                });
            }
        }
    }
}

// ── All benchmark groups ──────────────────────────────────────────────────────

fn benchmark_filter_steps() {
    let image_cases = discover_image_cases();
    bench_build_image_color_model_step(&image_cases);
    bench_build_image_grouping_step(&image_cases);

    let _ = std::fs::create_dir_all("target/bench-artifacts");
    let paths = sorted_image_paths();
    let total_cases = paths.len() * SEED_PREPROCESSING_CONFIGS.len();
    let mut done = 0;

    let seed_meta = |case: &PreparedCase| {
        let p = case.preprocessing;
        (
            p.color_model_label().to_string(),
            p.pixel_grouping_label(),
            p.group_transform_label().to_string(),
            p.grouped_pixels().to_string(),
        )
    };

    for path in paths {
        for &pre in SEED_PREPROCESSING_CONFIGS {
            done += 1;
            let file_name = path.file_name().unwrap().to_string_lossy();
            eprintln!(
                "[{done}/{total_cases}] preparing {file_name} ({})",
                pre.artifact_label()
            );
            let case = prepare_case(path.clone(), pre);
            let cases = std::slice::from_ref(&case);

            bench_step_group(
                "Entropy",
                cases,
                &ENTROPY_IMPLS,
                EntropyImpl::label,
                |_, case| case.bit_data_seed.clone(),
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "SelectBases",
                cases,
                &SELECT_BASES_IMPLS,
                SelectBasesImpl::label,
                |_, case| case.entropy_seed.clone(),
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "BuildBaseTable",
                cases,
                &BUILD_BASE_TABLE_IMPLS,
                BuildBaseTableImpl::label,
                |impl_, case| impl_.select_bases(case.entropy_seed.clone()),
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "EncodeData",
                cases,
                &ENCODE_IMPLS,
                EncodeImpl::label,
                |_, case| EncodeSeed {
                    pre_encode: case.base_table_ctx_seed.clone().unwrap(),
                },
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "DeltaEncodeBaseTable",
                cases,
                &DELTA_ENCODE_IMPLS,
                DeltaEncodeImpl::label,
                |_, case| case.pre_delta_compressed_seed.clone().unwrap(),
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| {
                    let raw = case
                        .pre_delta_compressed_seed
                        .as_ref()
                        .unwrap()
                        .base_table
                        .as_raw();
                    let bits_per_row = raw.first().map(|(bv, _)| bv.len()).unwrap_or(0);
                    (raw.len() * bits_per_row).div_ceil(8) as u64
                },
            );

            bench_step_group(
                "DecodeDeltaBaseTable",
                cases,
                &DECODE_DELTA_IMPLS,
                DecodeDeltaImpl::label,
                |impl_, case| match impl_ {
                    DecodeDeltaImpl::Unary => case.delta_unary_seed.clone().unwrap(),
                    DecodeDeltaImpl::Fixed => case.delta_fixed_seed.clone().unwrap(),
                },
                |_, input| DecodeDeltaBaseTable {}.process(input),
                seed_meta,
                |case| {
                    let raw = case
                        .pre_delta_compressed_seed
                        .as_ref()
                        .unwrap()
                        .base_table
                        .as_raw();
                    let bits_per_row = raw.first().map(|(bv, _)| bv.len()).unwrap_or(0);
                    (raw.len() * bits_per_row).div_ceil(8) as u64
                },
            );

            bench_step_group(
                "SaveIgdFile",
                cases,
                &SAVE_IGD_IMPLS,
                SaveIgdImpl::label,
                |impl_, case| match impl_ {
                    SaveIgdImpl::Normal => (
                        case.compressed_seeds.as_ref().unwrap().normal.clone(),
                        case.igd_paths.normal.clone(),
                    ),
                    SaveIgdImpl::RleOffset => (
                        case.compressed_seeds.as_ref().unwrap().rle_offset.clone(),
                        case.igd_paths.rle_offset.clone(),
                    ),
                    SaveIgdImpl::Huffman => (
                        case.compressed_seeds.as_ref().unwrap().huffman.clone(),
                        case.igd_paths.huffman.clone(),
                    ),
                },
                |implementation, (compressed, path)| implementation.process(compressed, path),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "LoadIgdFile",
                cases,
                &LOAD_IGD_IMPLS,
                LoadIgdImpl::label,
                |impl_, case| match impl_ {
                    LoadIgdImpl::Normal => case.igd_paths.normal.clone(),
                    LoadIgdImpl::RleOffset => case.igd_paths.rle_offset.clone(),
                    LoadIgdImpl::Huffman => case.igd_paths.huffman.clone(),
                },
                |implementation, path| implementation.process(path),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "DecompressFileData",
                cases,
                &DECOMPRESS_FILE_IMPLS,
                DecompressFileImpl::label,
                |impl_, case| match impl_ {
                    DecompressFileImpl::Normal => case
                        .loaded_compressed_seeds
                        .as_ref()
                        .unwrap()
                        .normal
                        .clone(),
                    DecompressFileImpl::RleOffset => case
                        .loaded_compressed_seeds
                        .as_ref()
                        .unwrap()
                        .rle_offset
                        .clone(),
                    DecompressFileImpl::Huffman => case
                        .loaded_compressed_seeds
                        .as_ref()
                        .unwrap()
                        .huffman
                        .clone(),
                },
                |implementation, input| implementation.process(input),
                seed_meta,
                |case| case.source_size,
            );

            bench_step_group(
                "DecompressRowsData",
                cases,
                &DECOMPRESS_ROWS_IMPLS,
                DecompressRowsImpl::label,
                |impl_, case| match impl_ {
                    DecompressRowsImpl::Normal => (
                        case.rows_context_seeds.as_ref().unwrap().normal.0.clone(),
                        case.rows_context_seeds.as_ref().unwrap().normal.1.clone(),
                    ),
                    DecompressRowsImpl::RleOffset => (
                        case.rows_context_seeds
                            .as_ref()
                            .unwrap()
                            .rle_offset
                            .0
                            .clone(),
                        case.rows_context_seeds
                            .as_ref()
                            .unwrap()
                            .rle_offset
                            .1
                            .clone(),
                    ),
                    DecompressRowsImpl::Huffman => (
                        case.rows_context_seeds.as_ref().unwrap().huffman.0.clone(),
                        case.rows_context_seeds.as_ref().unwrap().huffman.1.clone(),
                    ),
                },
                |implementation, (handle, indices)| implementation.process(handle, indices),
                seed_meta,
                |case| case.source_size,
            );
        }
    }
}

// ── CSV output ────────────────────────────────────────────────────────────────

fn write_bench_csv() {
    let records = BENCH_RECORDS.lock().unwrap();
    if records.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all("target/bench-results");
    let path = format!(
        "target/bench-results/performance_{}.csv",
        IMAGE_DIR.split('/').last().unwrap_or("results")
    );
    let mut out = String::from(
        "stage,impl,file,sample_time_ns,throughput_bytes_s,color_model_seed,pixel_grouping_seed,group_transform_seed,grouped_pixels_seed\n",
    );
    for r in records.iter() {
        let throughput = if r.sample_time_ns > 0 {
            r.source_bytes as u128 * 1_000_000_000 / r.sample_time_ns
        } else {
            0
        };
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            r.stage,
            r.impl_name,
            r.file,
            r.sample_time_ns,
            throughput,
            r.color_model_seed,
            r.pixel_grouping_seed,
            r.group_transform_seed,
            r.grouped_pixels_seed,
        ));
    }
    std::fs::write(&path, out).unwrap_or_else(|e| eprintln!("failed to write {path}: {e}"));
    println!("bench results written to {path} ({} rows)", records.len());
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    benchmark_filter_steps();
    write_bench_csv();
}
