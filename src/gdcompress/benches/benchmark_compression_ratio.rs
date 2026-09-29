use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::compression::encoding::BaseTable;
use gdcompress::prelude::*;
use gdcompress::{
    BaseSelectionContext, CompressedData, EntroGdError, EntropyScoredContext, IgdFile,
    PreEncodeContext,
};

// ── CSV collection ────────────────────────────────────────────────────────────

struct BenchRecord {
    group: &'static str,
    impl_name: String,
    file: String,
    color_model_seed: String,
    pixel_grouping_seed: String,
    group_transform_seed: String,
    grouped_pixels_seed: String,
    source_bytes: u64,
    total_compressed_bytes: u64,
    base_table_bytes: u64,
    deviation_stream_bytes: u64,
    parameters_bytes: u64,
    // Filled only for the select_bases group; empty string otherwise.
    base_table_compression_ratio: String,
}

static BENCH_RECORDS: LazyLock<Mutex<Vec<BenchRecord>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ── Canonical pipeline config ─────────────────────────────────────────────────

const NORMAL_PATIENCE: usize = 10;
const THRESHOLD_PATIENCE: usize = 5;
const ADAPTIVE_PATIENCE: usize = 5;
const ENTROPY_THRESHOLD: f64 = 0.75;
const WEIGHT_DECAY: f64 = 0.5;
const IMAGE_DIR: &str = "data/bench_datasets/kodak_dataset";

// ── Seed preprocessing ────────────────────────────────────────────────────────

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

    fn process(self, path: PathBuf) -> Result<gdcompress::BitDataSet, EntroGdError> {
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

// ── Seed preprocessing configs ────────────────────────────────────────────────
// Add / remove / edit rows here to change which preprocessing configs seed the
// downstream benchmark stages

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
    (Rgb, 1, 1, Raw),
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

// ── Size computation ──────────────────────────────────────────────────────────

struct CompressionSizes {
    total_compressed_bytes: u64,
    base_table_bytes: u64,
    deviation_stream_bytes: u64,
    parameters_bytes: u64,
}

// Returns raw_base_table_bytes / delta_base_table_bytes for a delta-encoded base table,
// or an empty string if the base table is not delta-encoded.
fn base_table_delta_ratio(compressed: &CompressedData) -> String {
    match &compressed.base_table {
        BaseTable::Delta(delta) => {
            let delta_bits = delta.first_sort_key.len() + delta.delta_bit_stream.len();
            if delta_bits == 0 {
                return String::new();
            }
            let raw_base_table = delta.decode_rows().unwrap_or_default();
            let raw_bits = raw_base_table.iter().map(|(bv, _)| bv.len()).sum::<usize>();
            let raw_bytes = raw_bits.div_ceil(8);
            let delta_bytes = delta_bits.div_ceil(8);
            format!("{:.4}", raw_bytes as f64 / delta_bytes as f64)
        }
        BaseTable::Raw(_) => String::new(),
    }
}

fn compute_sizes(compressed: &CompressedData) -> Result<CompressionSizes, EntroGdError> {
    let igd = IgdFile::from_compressed_data(compressed.clone())?;
    let total = igd.as_bytes().len() as u64;

    let base_table_bits = match &compressed.base_table {
        BaseTable::Raw(rows) => rows
            .first()
            .map(|(bv, _)| rows.len() * bv.len())
            .unwrap_or(0),
        BaseTable::Delta(delta) => delta.first_sort_key.len() + delta.delta_bit_stream.len(),
    };

    let stream_bits = compressed.encoded_data.get_encoded_size();
    let base_table_bytes = ((base_table_bits + 7) / 8) as u64;
    let stream_bytes = ((stream_bits + 7) / 8) as u64;
    let parameters_bytes = total.saturating_sub(base_table_bytes + stream_bytes);

    Ok(CompressionSizes {
        total_compressed_bytes: total,
        base_table_bytes,
        deviation_stream_bytes: stream_bytes,
        parameters_bytes,
    })
}

// ── Image discovery ───────────────────────────────────────────────────────────

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

// ── Group 1: select_bases ─────────────────────────────────────────────────────
//
// Isolates the effect of base selection strategy and counting method.
// Fixed: EncodeDataFusedDictionary (works with all BaseBit impls incl. HyperLogLog), no delta encoding.

#[derive(Clone, Copy)]
enum SelectBasesVariant {
    Naive,
    ThresholdNaive,
    ThresholdHll,
    AdaptiveNaive,
    AdaptiveHll,
}

impl SelectBasesVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::ThresholdNaive => "threshold_naive",
            Self::ThresholdHll => "threshold_hll",
            Self::AdaptiveNaive => "adaptive_naive",
            Self::AdaptiveHll => "adaptive_hll",
        }
    }

    fn process(self, ctx: EntropyScoredContext) -> Result<BaseSelectionContext, EntroGdError> {
        match self {
            Self::Naive => SelectBases {
                patience: NORMAL_PATIENCE,
            }
            .process(ctx),
            Self::ThresholdNaive => SelectBasesThreshold {
                patience: THRESHOLD_PATIENCE,
                base_bit_impl: BaseBitImpl::Naive,
                entropy_threshold: ENTROPY_THRESHOLD,
            }
            .process(ctx),
            Self::ThresholdHll => SelectBasesThreshold {
                patience: THRESHOLD_PATIENCE,
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
                entropy_threshold: ENTROPY_THRESHOLD,
            }
            .process(ctx),
            Self::AdaptiveNaive => SelectBasesAdaptive {
                width_decay: WEIGHT_DECAY,
                patience: ADAPTIVE_PATIENCE,
                base_bit_impl: BaseBitImpl::Naive,
            }
            .process(ctx),
            Self::AdaptiveHll => SelectBasesAdaptive {
                width_decay: WEIGHT_DECAY,
                patience: ADAPTIVE_PATIENCE,
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
            }
            .process(ctx),
        }
    }
}

const SELECT_BASES_VARIANTS: [SelectBasesVariant; 5] = [
    SelectBasesVariant::Naive,
    SelectBasesVariant::ThresholdNaive,
    SelectBasesVariant::ThresholdHll,
    SelectBasesVariant::AdaptiveNaive,
    SelectBasesVariant::AdaptiveHll,
];

fn bench_select_bases(paths: &[PathBuf]) {
    let total = paths.len() * SEED_PREPROCESSING_CONFIGS.len() * SELECT_BASES_VARIANTS.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        for pre in SEED_PREPROCESSING_CONFIGS {
            let bit_data = match pre.process(path.clone()) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("skip {file_name}: {e}");
                    continue;
                }
            };
            let source_bytes = (bit_data.data.num_rows * bit_data.data.chunk_size / 8) as u64;
            let entropy_ctx = Entropy {}.process(bit_data).unwrap();

            for variant in SELECT_BASES_VARIANTS {
                done += 1;
                eprintln!(
                    "[{done}/{total}] select_bases/{}/{}",
                    variant.name(),
                    file_name
                );

                let base_sel = variant.process(entropy_ctx.clone()).unwrap();
                let pre_encode = BuildBaseTable {}.process(base_sel).unwrap();
                let compressed = EncodeData {}.process(pre_encode).unwrap();
                let sizes = compute_sizes(&compressed).unwrap();

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: "select_bases",
                    impl_name: variant.name().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    base_table_compression_ratio: String::new(),
                });
            }
        }
    }
}

// ── Group 2: encoding ─────────────────────────────────────────────────────────
//
// Isolates the effect of the deviation-stream encoding scheme.
// Fixed: naive SelectBases, BuildBaseTable, no delta encoding.

#[derive(Clone, Copy)]
enum EncodingVariant {
    Normal,
    RleOffset,
    Huffman,
}

impl EncodingVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, pre_encode: PreEncodeContext) -> Result<CompressedData, EntroGdError> {
        match self {
            Self::Normal => EncodeData {}.process(pre_encode),
            Self::RleOffset => EncodeDataOffsetRLE {}.process(pre_encode),
            Self::Huffman => EncodeDataHuffman {}.process(pre_encode),
        }
    }
}

const ENCODING_VARIANTS: [EncodingVariant; 3] = [
    EncodingVariant::Normal,
    EncodingVariant::RleOffset,
    EncodingVariant::Huffman,
];

fn bench_encoding(paths: &[PathBuf]) {
    let total = paths.len() * SEED_PREPROCESSING_CONFIGS.len() * ENCODING_VARIANTS.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        for pre in SEED_PREPROCESSING_CONFIGS {
            let bit_data = match pre.process(path.clone()) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("skip {file_name}: {e}");
                    continue;
                }
            };
            let source_bytes = (bit_data.data.num_rows * bit_data.data.chunk_size / 8) as u64;
            let entropy_ctx = Entropy {}.process(bit_data).unwrap();
            let base_sel = SelectBases {
                patience: NORMAL_PATIENCE,
            }
            .process(entropy_ctx)
            .unwrap();
            let pre_encode_base = BuildBaseTable {}.process(base_sel).unwrap();

            for variant in ENCODING_VARIANTS {
                done += 1;
                eprintln!("[{done}/{total}] encoding/{}/{}", variant.name(), file_name);

                let compressed = variant.process(pre_encode_base.clone()).unwrap();
                let sizes = compute_sizes(&compressed).unwrap();

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: "encoding",
                    impl_name: variant.name().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    base_table_compression_ratio: String::new(),
                });
            }
        }
    }
}

// ── Group 3: delta_encoding ───────────────────────────────────────────────────
//
// Isolates the effect of base-table delta encoding.
// Fixed: naive SelectBases, EncodeData.
// Delta encoding requires a sorted base table (BuildSortedBaseTable); without it
// DeltaEncodeBaseTable is a no-op, so only Raw vs. Delta are meaningful variants.

#[derive(Clone, Copy)]
enum DeltaVariant {
    Raw,
    Delta,
    DeltaFixed,
}

impl DeltaVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Delta => "delta",
            Self::DeltaFixed => "delta_fixed",
        }
    }

    fn process(self, entropy_ctx: EntropyScoredContext) -> Result<CompressedData, EntroGdError> {
        let base_sel = SelectBases {
            patience: NORMAL_PATIENCE,
        }
        .process(entropy_ctx)?;
        match self {
            Self::Raw => {
                let pre = BuildBaseTable {}.process(base_sel)?;
                EncodeData {}.process(pre)
            }
            Self::Delta => {
                let pre = BuildSortedBaseTable {}.process(base_sel)?;
                let compressed = EncodeData {}.process(pre)?;
                DeltaEncodeBaseTable {}.process(compressed)
            }
            Self::DeltaFixed => {
                let pre = BuildSortedBaseTable {}.process(base_sel)?;
                let compressed = EncodeData {}.process(pre)?;
                DeltaEncodeBaseTableFixed {}.process(compressed)
            }
        }
    }
}

const DELTA_VARIANTS: [DeltaVariant; 3] = [
    DeltaVariant::Raw,
    DeltaVariant::Delta,
    DeltaVariant::DeltaFixed,
];

fn bench_delta_encoding(paths: &[PathBuf]) {
    let total = paths.len() * SEED_PREPROCESSING_CONFIGS.len() * DELTA_VARIANTS.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        for pre in SEED_PREPROCESSING_CONFIGS {
            let bit_data = match pre.process(path.clone()) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("skip {file_name}: {e}");
                    continue;
                }
            };
            let source_bytes = (bit_data.data.num_rows * bit_data.data.chunk_size / 8) as u64;
            let entropy_ctx = Entropy {}.process(bit_data).unwrap();

            for variant in DELTA_VARIANTS {
                done += 1;
                eprintln!(
                    "[{done}/{total}] delta_encoding/{}/{}",
                    variant.name(),
                    file_name
                );

                let compressed = variant.process(entropy_ctx.clone()).unwrap();
                let sizes = compute_sizes(&compressed).unwrap();
                let bt_delta_ratio = match variant {
                    DeltaVariant::Delta | DeltaVariant::DeltaFixed => {
                        let r = base_table_delta_ratio(&compressed);
                        if r.is_empty() {
                            "1.0000".to_string()
                        } else {
                            r
                        }
                    }
                    DeltaVariant::Raw => String::new(),
                };

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: "delta_encoding",
                    impl_name: variant.name().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    base_table_compression_ratio: bt_delta_ratio,
                });
            }
        }
    }
}

// ── Group 4: color_model ──────────────────────────────────────────────────────
//
// Isolates the effect of color model on compression ratio.
// Fixed: 1x1 pixel grouping, Raw transform, naive SelectBases, EncodeData.

#[derive(Clone, Copy)]
enum ColorModelVariant {
    Rgb,
    YCoCgR,
}

impl ColorModelVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::YCoCgR => "ycocgr",
        }
    }

    fn color_model(self) -> ImageColorModel {
        match self {
            Self::Rgb => ImageColorModel::Rgb,
            Self::YCoCgR => ImageColorModel::YCoCgR,
        }
    }
}

const COLOR_MODEL_VARIANTS: [ColorModelVariant; 2] =
    [ColorModelVariant::Rgb, ColorModelVariant::YCoCgR];

fn bench_color_model(paths: &[PathBuf]) {
    let total = paths.len() * SEED_PREPROCESSING_CONFIGS.len() * COLOR_MODEL_VARIANTS.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        for seed in SEED_PREPROCESSING_CONFIGS {
            for variant in COLOR_MODEL_VARIANTS {
                done += 1;
                eprintln!(
                    "[{done}/{total}] color_model/{}/{}",
                    variant.name(),
                    file_name
                );

                // Override the seed's color_model with the variant's so each color
                // model is measured against the full grouping/transform grid.
                let pre = SeedPreprocessing {
                    color_model: variant.color_model(),
                    group_w: seed.group_w,
                    group_h: seed.group_h,
                    grouping_transform: seed.grouping_transform,
                };
                let bit_data = match pre.process(path.clone()) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("skip {file_name}: {e}");
                        continue;
                    }
                };
                let source_bytes = (bit_data.data.num_rows * bit_data.data.chunk_size / 8) as u64;
                let entropy_ctx = Entropy {}.process(bit_data).unwrap();
                let base_sel = SelectBases {
                    patience: NORMAL_PATIENCE,
                }
                .process(entropy_ctx)
                .unwrap();
                let pre_encode = BuildBaseTable {}.process(base_sel).unwrap();
                let compressed = EncodeData {}.process(pre_encode).unwrap();
                let sizes = compute_sizes(&compressed).unwrap();

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: "color_model",
                    impl_name: variant.name().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    base_table_compression_ratio: String::new(),
                });
            }
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
    let dataset = IMAGE_DIR.split('/').last().unwrap_or("results");
    let path = format!("target/bench-results/compression_ratio_{dataset}.csv");
    let mut out = String::from(
        "group,impl_name,file,color_model_seed,pixel_grouping_seed,group_transform_seed,grouped_pixels_seed,source_bytes,total_compressed_bytes,base_table_bytes,deviation_stream_bytes,parameters_bytes,compression_ratio,base_table_compression_ratio\n",
    );
    for r in records.iter() {
        let compression_ratio = if r.total_compressed_bytes > 0 {
            format!(
                "{:.4}",
                r.source_bytes as f64 / r.total_compressed_bytes as f64
            )
        } else {
            String::from("0.0000")
        };
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.group,
            r.impl_name,
            r.file,
            r.color_model_seed,
            r.pixel_grouping_seed,
            r.group_transform_seed,
            r.grouped_pixels_seed,
            r.source_bytes,
            r.total_compressed_bytes,
            r.base_table_bytes,
            r.deviation_stream_bytes,
            r.parameters_bytes,
            compression_ratio,
            r.base_table_compression_ratio,
        ));
    }
    std::fs::write(&path, out).unwrap_or_else(|e| eprintln!("failed to write {path}: {e}"));
    println!(
        "compression ratio results written to {path} ({} rows)",
        records.len()
    );
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let paths = sorted_image_paths();
    bench_select_bases(&paths);
    bench_encoding(&paths);
    bench_delta_encoding(&paths);
    bench_color_model(&paths);
    write_bench_csv();
}
