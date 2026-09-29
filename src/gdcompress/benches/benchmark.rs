use std::hint::black_box;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use png::{
    BitDepth, ColorType as PngColorType, Compression as PngCompression, Encoder as PngEncoder,
    Filter as PngFilter,
};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::compression::encoding::BaseTable;
use gdcompress::prelude::*;
use gdcompress::{BitDataSet, CompressedData, DecompressRandomAccessHandle, EntroGdError, IgdFile};

// ── CSV collection ────────────────────────────────────────────────────────────

struct BenchRecord {
    group: &'static str,
    pipeline: String,
    file: String,
    // EntroGD preprocessing metadata (empty for external codecs)
    color_model_seed: String,
    pixel_grouping_seed: String,
    group_transform_seed: String,
    grouped_pixels_seed: String,
    source_bytes: u64,
    sample_time_ns: u128,
    // Compression size breakdown (0 for external codecs and random_access)
    total_compressed_bytes: u64,
    base_table_bytes: u64,
    deviation_stream_bytes: u64,
    parameters_bytes: u64,
    // Random access metadata (0/"" for compress and decompress_file groups)
    batch_size: usize,
    num_rows: usize,
    access_pattern: String,
}

static BENCH_RECORDS: LazyLock<Mutex<Vec<BenchRecord>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ── Harness config ────────────────────────────────────────────────────────────

const N_RUNS: usize = 3;
const DEFAULT_DATA_ROOT: &str = "data/bench_datasets";

// ── Pipeline constants ────────────────────────────────────────────────────────

const PATIENCE_BEST: usize = 10;
const PATIENCE_FAST: usize = 5;
const WIDTH_DECAY_DEFAULT: f64 = 0.5;

// ── Preprocessing spec ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct PreprocessingSpec {
    color_model: ImageColorModel,
    group_w: u32,
    group_h: u32,
    grouping_transform: ImageGroupingTransform,
}

impl PreprocessingSpec {
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

    fn process(self, image: image::DynamicImage) -> Result<BitDataSet, EntroGdError> {
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

// ── EntroGD compression pipelines ─────────────────────────────────────────────

#[derive(Clone, Copy)]
enum CompressionPipeline {
    ImageBestRatio,
    ImageBestRatioRle,
    ImageFast,
}

impl CompressionPipeline {
    fn label(self) -> &'static str {
        match self {
            Self::ImageBestRatio => "image_best_ratio",
            Self::ImageBestRatioRle => "image_best_ratio_rle",
            Self::ImageFast => "image_fast",
        }
    }

    fn preprocessing(self) -> PreprocessingSpec {
        match self {
            Self::ImageBestRatio => PreprocessingSpec {
                color_model: ImageColorModel::YCoCgR,
                group_w: 2,
                group_h: 2,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
            },
            Self::ImageBestRatioRle => PreprocessingSpec {
                color_model: ImageColorModel::YCoCgR,
                group_w: 2,
                group_h: 2,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
            },
            Self::ImageFast => PreprocessingSpec {
                color_model: ImageColorModel::YCoCgR,
                group_w: 3,
                group_h: 2,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
            },
        }
    }

    fn run(self, bit_data: BitDataSet) -> Result<CompressedData, EntroGdError> {
        match self {
            Self::ImageBestRatio => Entropy {}
                .then(SelectBases {
                    patience: PATIENCE_BEST,
                })
                .then(BuildSortedBaseTable {})
                .then(EncodeDataHuffman {})
                .then(DeltaEncodeBaseTableFixed {})
                .process(bit_data),
            Self::ImageBestRatioRle => Entropy {}
                .then(SelectBases {
                    patience: PATIENCE_BEST,
                })
                .then(BuildSortedBaseTable {})
                .then(EncodeDataOffsetRLE {})
                .then(DeltaEncodeBaseTableFixed {})
                .process(bit_data),
            Self::ImageFast => Entropy {}
                .then(SelectBasesAdaptive {
                    patience: PATIENCE_FAST,
                    width_decay: WIDTH_DECAY_DEFAULT,
                    base_bit_impl: BaseBitImpl::HyperLogLogCount,
                })
                .then(BuildBaseTable {})
                .then(EncodeData {})
                .process(bit_data),
        }
    }
}

const PIPELINES: [CompressionPipeline; 3] = [
    CompressionPipeline::ImageBestRatio,
    CompressionPipeline::ImageBestRatioRle,
    CompressionPipeline::ImageFast,
];

// ── External codecs ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum ExternalCodec {
    Png,
    WebPLossless,
    Qoi,
    JpegLs,
}

impl ExternalCodec {
    fn label(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::WebPLossless => "webp_lossless",
            Self::Qoi => "qoi",
            Self::JpegLs => "jpeg_ls",
        }
    }

    fn compress(self, rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        match self {
            Self::Png => {
                let mut out = Vec::new();
                let mut encoder = PngEncoder::new(&mut out, width, height);
                encoder.set_color(PngColorType::Rgba);
                encoder.set_depth(BitDepth::Eight);
                encoder.set_compression(PngCompression::Balanced);
                encoder.set_filter(PngFilter::Adaptive);
                let mut writer = encoder.write_header().unwrap();
                writer.write_image_data(rgba).unwrap();
                drop(writer);
                out
            }
            Self::WebPLossless => {
                let img = image::RgbaImage::from_raw(width, height, rgba.to_vec()).unwrap();
                let dyn_img = image::DynamicImage::ImageRgba8(img);
                let mut buf = Cursor::new(Vec::new());
                dyn_img
                    .write_to(&mut buf, image::ImageFormat::WebP)
                    .unwrap();
                buf.into_inner()
            }
            Self::Qoi => qoi::encode_to_vec(rgba, width, height).unwrap(),
            Self::JpegLs => {
                use dicom_toolkit_codec::jpeg_ls::JpegLsCodec;
                JpegLsCodec::encode_frame(rgba, width, height, 8, 4, 0).unwrap()
            }
        }
    }

    fn decompress(self, data: &[u8]) -> Vec<u8> {
        match self {
            Self::Png => {
                let img =
                    image::load_from_memory_with_format(data, image::ImageFormat::Png).unwrap();
                img.to_rgba8().into_raw()
            }
            Self::WebPLossless => {
                let img =
                    image::load_from_memory_with_format(data, image::ImageFormat::WebP).unwrap();
                img.to_rgba8().into_raw()
            }
            Self::Qoi => {
                let (_, pixels) = qoi::decode_to_vec(data).unwrap();
                pixels
            }
            Self::JpegLs => {
                use dicom_toolkit_codec::jpeg_ls::JpegLsCodec;
                JpegLsCodec::decode_frame(data).unwrap().pixels
            }
        }
    }
}

const EXTERNAL_CODECS: [ExternalCodec; 4] = [
    ExternalCodec::Png,
    ExternalCodec::WebPLossless,
    ExternalCodec::Qoi,
    ExternalCodec::JpegLs,
];

// ── Size computation ──────────────────────────────────────────────────────────

struct CompressionSizes {
    total_compressed_bytes: u64,
    base_table_bytes: u64,
    deviation_stream_bytes: u64,
    parameters_bytes: u64,
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
    let base_table_bytes = base_table_bits.div_ceil(8) as u64;
    let stream_bytes = stream_bits.div_ceil(8) as u64;
    let parameters_bytes = total.saturating_sub(base_table_bytes + stream_bytes);

    Ok(CompressionSizes {
        total_compressed_bytes: total,
        base_table_bytes,
        deviation_stream_bytes: stream_bytes,
        parameters_bytes,
    })
}

// ── Image discovery ───────────────────────────────────────────────────────────

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    let mut paths: Vec<PathBuf> = read_dir
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let ext = path.extension()?.to_string_lossy().to_lowercase();
            (ext == "png" || ext == "jpg" || ext == "jpeg").then_some(path)
        })
        .collect();
    paths.sort();
    paths
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("results")
        .to_string()
}

/// Returns one entry per dataset. If `root` contains images directly, that's the
/// only dataset; otherwise each immediate subdirectory containing images is a
/// dataset.
fn discover_datasets(root: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let direct = collect_images(root);
    if !direct.is_empty() {
        return vec![(dir_name(root), direct)];
    }

    let read_dir =
        std::fs::read_dir(root).unwrap_or_else(|e| panic!("cannot read {}: {}", root.display(), e));
    let mut subdirs: Vec<PathBuf> = read_dir
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            path.is_dir().then_some(path)
        })
        .collect();
    subdirs.sort();

    let mut datasets = Vec::new();
    for sub in subdirs {
        let imgs = collect_images(&sub);
        if !imgs.is_empty() {
            datasets.push((dir_name(&sub), imgs));
        }
    }
    datasets
}

// ── Random index generation ───────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// Returns `n` unique indices from [0, num_rows) in ascending order.
fn random_indices(n: usize, num_rows: usize, seed: u64) -> Vec<usize> {
    if n >= num_rows {
        return (0..num_rows).collect();
    }
    let mut pool: Vec<usize> = (0..num_rows).collect();
    let mut state = seed;
    for i in 0..n {
        let j = i + (xorshift64(&mut state) as usize % (num_rows - i));
        pool.swap(i, j);
    }
    let mut selected = pool[..n].to_vec();
    selected.sort_unstable();
    selected
}

fn sequential_indices(n: usize, num_rows: usize) -> Vec<usize> {
    (0..n.min(num_rows)).collect()
}

fn batch_sizes(num_rows: usize) -> [(usize, &'static str); 3] {
    [
        (1_usize.max(num_rows / 100), "1pct"),
        (1_usize.max(num_rows / 10), "10pct"),
        (1_usize.max(num_rows / 2), "50pct"),
    ]
}

// ── Benchmark 1: EntroGD compression core ────────────────────────────────────

fn bench_compression_core(paths: &[PathBuf]) {
    let stage = "compress";
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage.contains(filter.as_str()) {
        return;
    }

    let total = paths.len() * PIPELINES.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        let image = match OpenImage.process(path.clone()) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skip {file_name}: {e}");
                continue;
            }
        };

        for &pipeline in &PIPELINES {
            done += 1;
            eprintln!(
                "[{done}/{total}] generalized deduplication {stage}/{}/{file_name}",
                pipeline.label()
            );

            let pre = pipeline.preprocessing();

            // Warmup run; also extracts source_bytes from the resulting BitDataSet.
            let warmup_bd = pre.process(image.clone()).unwrap();
            let source_bytes = (warmup_bd.info.original_size_bits() / 8) as u64;
            let _ = black_box(pipeline.run(warmup_bd));

            for _ in 0..N_RUNS {
                let image_clone = image.clone();
                let start = Instant::now();
                let bit_data = pre.process(image_clone).unwrap();
                let compressed = pipeline.run(bit_data).unwrap();
                let elapsed = start.elapsed();
                black_box(&compressed);

                let sizes = compute_sizes(&compressed).unwrap();

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: stage,
                    pipeline: pipeline.label().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    sample_time_ns: elapsed.as_nanos(),
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    batch_size: 0,
                    num_rows: 0,
                    access_pattern: String::new(),
                });
            }
        }
    }
}

// ── Benchmark 2: EntroGD decompress file ─────────────────────────────────────

fn bench_decompress_file(paths: &[PathBuf]) {
    let stage = "decompress_file";
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage.contains(filter.as_str()) {
        return;
    }

    let total = paths.len() * PIPELINES.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        let image = match OpenImage.process(path.clone()) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skip {file_name}: {e}");
                continue;
            }
        };

        for &pipeline in &PIPELINES {
            done += 1;
            eprintln!(
                "[{done}/{total}] generalized deduplication {stage}/{}/{file_name}",
                pipeline.label()
            );

            let pre = pipeline.preprocessing();
            let bit_data = pre.process(image.clone()).unwrap();
            let source_bytes = (bit_data.info.original_size_bits() / 8) as u64;
            let compressed = pipeline.run(bit_data).unwrap();
            let sizes = compute_sizes(&compressed).unwrap();

            // warmup
            let _ = black_box(DecompressFileData {}.process(compressed.clone()));

            for _ in 0..N_RUNS {
                let compressed_clone = compressed.clone();
                let start = Instant::now();
                let decompressed = DecompressFileData {}.process(compressed_clone).unwrap();
                let elapsed = start.elapsed();
                black_box(decompressed);

                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    group: stage,
                    pipeline: pipeline.label().to_string(),
                    file: file_name.clone(),
                    color_model_seed: pre.color_model_label().to_string(),
                    pixel_grouping_seed: pre.pixel_grouping_label(),
                    group_transform_seed: pre.group_transform_label().to_string(),
                    grouped_pixels_seed: pre.grouped_pixels().to_string(),
                    source_bytes,
                    sample_time_ns: elapsed.as_nanos(),
                    total_compressed_bytes: sizes.total_compressed_bytes,
                    base_table_bytes: sizes.base_table_bytes,
                    deviation_stream_bytes: sizes.deviation_stream_bytes,
                    parameters_bytes: sizes.parameters_bytes,
                    batch_size: 0,
                    num_rows: 0,
                    access_pattern: String::new(),
                });
            }
        }
    }
}

// ── Benchmark 3: EntroGD random access ───────────────────────────────────────

fn bench_random_access(paths: &[PathBuf]) {
    let stage = "random_access";
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage.contains(filter.as_str()) {
        return;
    }

    let total = paths.len() * PIPELINES.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        let image = match OpenImage.process(path.clone()) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skip {file_name}: {e}");
                continue;
            }
        };

        for &pipeline in &PIPELINES {
            done += 1;
            eprintln!(
                "[{done}/{total}] generalized deduplication {stage}/{}/{file_name}",
                pipeline.label()
            );

            let pre = pipeline.preprocessing();
            let bit_data = pre.process(image.clone()).unwrap();
            let num_rows = bit_data.data.num_rows;
            let source_bytes = (bit_data.info.original_size_bits() / 8) as u64;
            let compressed = pipeline.run(bit_data).unwrap();
            let sizes = compute_sizes(&compressed).unwrap();

            for (batch_size, batch_label) in batch_sizes(num_rows) {
                let random_idx = random_indices(batch_size, num_rows, 0xdeadbeef_cafebabe);
                let sequential_idx = sequential_indices(batch_size, num_rows);

                for (indices, pattern) in [
                    (random_idx.as_slice(), "random"),
                    (sequential_idx.as_slice(), "sequential"),
                ] {
                    // warmup
                    let _ = black_box(
                        DecompressRandomAccessHandle::new(compressed.clone())
                            .and_then(|h| h.decompress_samples(indices)),
                    );

                    for _ in 0..N_RUNS {
                        let compressed_clone = compressed.clone();
                        let start = Instant::now();
                        let handle = DecompressRandomAccessHandle::new(compressed_clone).unwrap();
                        let decompressed = handle.decompress_samples(indices).unwrap();
                        let elapsed = start.elapsed();
                        black_box(decompressed);

                        BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                            group: stage,
                            pipeline: pipeline.label().to_string(),
                            file: file_name.clone(),
                            color_model_seed: pre.color_model_label().to_string(),
                            pixel_grouping_seed: pre.pixel_grouping_label(),
                            group_transform_seed: pre.group_transform_label().to_string(),
                            grouped_pixels_seed: pre.grouped_pixels().to_string(),
                            source_bytes,
                            sample_time_ns: elapsed.as_nanos(),
                            total_compressed_bytes: sizes.total_compressed_bytes,
                            base_table_bytes: sizes.base_table_bytes,
                            deviation_stream_bytes: sizes.deviation_stream_bytes,
                            parameters_bytes: sizes.parameters_bytes,
                            batch_size,
                            num_rows,
                            access_pattern: format!("{batch_label}_{pattern}"),
                        });
                    }
                }
            }
        }
    }
}

// ── Benchmark 4: external codec compress + decompress ────────────────────────

fn bench_external(paths: &[PathBuf]) {
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();

    let total = paths.len() * EXTERNAL_CODECS.len();
    let mut done = 0;

    for path in paths {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        let image = match OpenImage.process(path.clone()) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skip {file_name}: {e}");
                continue;
            }
        };

        let rgba8 = image.to_rgba8();
        let (width, height) = rgba8.dimensions();
        let raw = rgba8.into_raw();
        let source_bytes = raw.len() as u64;

        for &codec in &EXTERNAL_CODECS {
            done += 1;

            // ── compress ──────────────────────────────────────────────────────
            let compress_stage = "compress";
            if filter.is_empty() || compress_stage.contains(filter.as_str()) {
                eprintln!(
                    "[{done}/{total}] external {compress_stage}/{}/{file_name}",
                    codec.label()
                );

                // Warmup + produce the compressed seed used for decompression below.
                let compressed_seed = codec.compress(&raw, width, height);
                let compressed_bytes = compressed_seed.len() as u64;

                for _ in 0..N_RUNS {
                    let start = Instant::now();
                    let out = codec.compress(&raw, width, height);
                    let elapsed = start.elapsed();
                    black_box(out);

                    BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                        group: compress_stage,
                        pipeline: codec.label().to_string(),
                        file: file_name.clone(),
                        color_model_seed: String::new(),
                        pixel_grouping_seed: String::new(),
                        group_transform_seed: String::new(),
                        grouped_pixels_seed: String::new(),
                        source_bytes,
                        sample_time_ns: elapsed.as_nanos(),
                        total_compressed_bytes: compressed_bytes,
                        base_table_bytes: 0,
                        deviation_stream_bytes: 0,
                        parameters_bytes: 0,
                        batch_size: 0,
                        num_rows: 0,
                        access_pattern: String::new(),
                    });
                }

                // ── decompress_file ───────────────────────────────────────────
                let decompress_stage = "decompress_file";
                if filter.is_empty() || decompress_stage.contains(filter.as_str()) {
                    // warmup
                    let _ = black_box(codec.decompress(&compressed_seed));

                    for _ in 0..N_RUNS {
                        let start = Instant::now();
                        let pixels = codec.decompress(&compressed_seed);
                        let elapsed = start.elapsed();
                        black_box(pixels);

                        BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                            group: decompress_stage,
                            pipeline: codec.label().to_string(),
                            file: file_name.clone(),
                            color_model_seed: String::new(),
                            pixel_grouping_seed: String::new(),
                            group_transform_seed: String::new(),
                            grouped_pixels_seed: String::new(),
                            source_bytes,
                            sample_time_ns: elapsed.as_nanos(),
                            total_compressed_bytes: compressed_bytes,
                            base_table_bytes: 0,
                            deviation_stream_bytes: 0,
                            parameters_bytes: 0,
                            batch_size: 0,
                            num_rows: 0,
                            access_pattern: String::new(),
                        });
                    }
                }
            }
        }
    }
}

// ── CSV output ────────────────────────────────────────────────────────────────

fn write_bench_csv(dataset: &str) {
    let records = BENCH_RECORDS.lock().unwrap();
    if records.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all("target/bench-results");
    let path = format!("target/bench-results/benchmark_{dataset}.csv");

    let mut out = String::from(
        "group,pipeline,file,\
         color_model_seed,pixel_grouping_seed,group_transform_seed,grouped_pixels_seed,\
         source_bytes,sample_time_ns,throughput_bytes_s,\
         total_compressed_bytes,base_table_bytes,deviation_stream_bytes,parameters_bytes,\
         compression_ratio,batch_size,access_pattern\n",
    );

    for r in records.iter() {
        let throughput_numerator = if r.group == "random_access" && r.num_rows > 0 {
            let pixels_per_group: u64 = r.grouped_pixels_seed.parse().unwrap_or(1);
            r.batch_size as u64 * pixels_per_group * 4
        } else {
            r.source_bytes
        };
        let throughput = if r.sample_time_ns > 0 {
            throughput_numerator as u128 * 1_000_000_000 / r.sample_time_ns
        } else {
            0
        };
        let compression_ratio = if r.total_compressed_bytes > 0 {
            format!(
                "{:.4}",
                r.source_bytes as f64 / r.total_compressed_bytes as f64
            )
        } else {
            String::new()
        };
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.group,
            r.pipeline,
            r.file,
            r.color_model_seed,
            r.pixel_grouping_seed,
            r.group_transform_seed,
            r.grouped_pixels_seed,
            r.source_bytes,
            r.sample_time_ns,
            throughput,
            r.total_compressed_bytes,
            r.base_table_bytes,
            r.deviation_stream_bytes,
            r.parameters_bytes,
            compression_ratio,
            r.batch_size,
            r.access_pattern,
        ));
    }

    std::fs::write(&path, out).unwrap_or_else(|e| eprintln!("failed to write {path}: {e}"));
    println!(
        "benchmark results written to {path} ({} rows)",
        records.len()
    );
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let root = std::env::var("BENCH_DATA_DIR").unwrap_or_else(|_| DEFAULT_DATA_ROOT.to_string());
    let root = PathBuf::from(root);
    let datasets = discover_datasets(&root);
    if datasets.is_empty() {
        eprintln!("no images found in {}", root.display());
        return;
    }

    for (i, (name, paths)) in datasets.iter().enumerate() {
        eprintln!(
            "=== dataset [{}/{}]: {} ({} images) ===",
            i + 1,
            datasets.len(),
            name,
            paths.len()
        );
        bench_compression_core(paths);
        bench_decompress_file(paths);
        bench_random_access(paths);
        bench_external(paths);
        write_bench_csv(name);
        BENCH_RECORDS.lock().unwrap().clear();
    }
}
