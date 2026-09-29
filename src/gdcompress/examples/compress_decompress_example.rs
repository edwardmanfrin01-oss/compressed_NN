use gdcompress::prelude::*;
use gdcompress::{
    EntroGdError, PreprocessOptions, ScopedTimer, init_logging, load_csv,
    reconstruct_feature_value, reconstruct_to_dataframe,
};
use polars::prelude::{CsvWriter, DataFrame, DataType, SerWriter};
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    unsafe {
        env::set_var("ENTRO_GD_LOG_TO_STDERR", "1");
        env::set_var("RUST_LOG", "debug");
    }
    let _log_handle = init_logging();
    let _timer = ScopedTimer::info("Total compression-decompression process");

    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "data/tabular/aarhus-citylab.csv".to_string()
    };

    let input = Path::new(&input_path);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");

    let tgd_path = Path::new("data/compressed").join(format!("{}.tgd", stem));
    let decompressed_path =
        Path::new("data/decompressed").join(format!("{}-decompressed.csv", stem));

    tracing::info!("Loading CSV: {}", input.display());

    let df = load_csv(input_path, true, None).map_err(|e| EntroGdError::DataLoad {
        message: format!("failed to load CSV: {}", e),
    })?;

    tracing::info!("Loaded: {} rows, {} columns", df.height(), df.width());

    let original_size_bytes = dataframe_memory_size_bytes(&df);

    let bit_data = BuildBitDataSet {
        options: PreprocessOptions::default(),
        pad_rows_to_word: false,
    }
    .process(df)?;

    let compression_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 100 })
        .then(SelectBasesAdaptive {
            width_decay: 0.5,
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
        })
        .then(BuildBaseTable {})
        .then(EncodeData {});

    fs::create_dir_all("data/compressed")?;
    fs::create_dir_all("data/decompressed")?;

    tracing::info!("Compressing...");
    let compressed = compression_pipeline.process(bit_data)?;

    tracing::info!("Saving TGD: {}", tgd_path.display());
    let compressed_path = SaveTgdFile {
        output_path: tgd_path,
    }
    .process(compressed.clone())?;

    let tgd_size_bytes = fs::metadata(&compressed_path)?.len() as usize;

    tracing::info!("Compression complete:");
    tracing::info!(
        "  Original file: {} bytes ({:.1} KB)",
        original_size_bytes,
        original_size_bytes as f64 / 1024.0
    );
    tracing::info!(
        "  Compressed file: {} bytes ({:.1} KB)",
        tgd_size_bytes,
        tgd_size_bytes as f64 / 1024.0
    );
    tracing::info!(
        "  Encoded stream: {} bits ({} Bytes)",
        compressed.encoded_data.get_encoded_size(),
        compressed.encoded_data.get_encoded_size() / 8
    );
    tracing::info!("  Base table entries: {}", compressed.base_table.len());
    tracing::info!(
        "  Base bit positions: {:?}",
        compressed.layout.selected_base_bit_positions()
    );

    if original_size_bytes > 0 && tgd_size_bytes > 0 {
        tracing::info!(
            "  Compression rate: {:.2}x",
            original_size_bytes as f64 / tgd_size_bytes as f64
        );
    }

    tracing::info!("Decompressing...");
    let compressed_data = LoadTgdFile {}.process(compressed_path.clone())?;
    let decompressed = DecompressFileData {}.process(compressed_data.clone())?;

    tracing::info!(
        "Decompressed: {} rows, {} features",
        decompressed.data.num_rows,
        decompressed.info.num_features()
    );

    if let Some(analytics) = (DecompressAnalytics {}).process(compressed_data.clone())? {
        tracing::debug!("Analytics: {} condensed samples", analytics.samples.len());
        for (i, (sample, weight)) in analytics
            .samples
            .iter()
            .zip(analytics.weights.iter())
            .enumerate()
        {
            tracing::trace!("  Sample {}: {} weight", i, weight);
            for feature_idx in 0..compressed_data.metadata.num_features() {
                let feature_start = compressed_data.metadata.feature_offset(feature_idx);
                let feature_end =
                    feature_start + compressed_data.metadata.feature_bits(feature_idx);
                let feature_bits = &sample[feature_start..feature_end];
                let spec = compressed_data.metadata.feature_spec(feature_idx);

                let formatted = reconstruct_feature_value(feature_bits, spec).to_string();
                tracing::trace!("    Feature {}: {}", feature_idx, formatted);
            }
        }
    }

    tracing::info!("Writing CSV: {}", decompressed_path.display());
    let mut df = reconstruct_to_dataframe(&decompressed)?;
    let mut f = std::fs::File::create(&decompressed_path)?;
    CsvWriter::new(&mut f).finish(&mut df)?;

    tracing::info!("Done.");
    Ok(())
}

fn dataframe_memory_size_bytes(df: &DataFrame) -> usize {
    df.columns()
        .iter()
        .map(|col: &polars::prelude::Column| {
            let n = col.len();
            match col.dtype() {
                DataType::Int8 | DataType::UInt8 => n,
                DataType::Int16 | DataType::UInt16 => n * 2,
                DataType::Int32 | DataType::UInt32 | DataType::Float32 => n * 4,
                DataType::Int64 | DataType::UInt64 | DataType::Float64 => n * 8,
                DataType::String => col
                    .str()
                    .map(|s: &polars::prelude::StringChunked| {
                        s.iter()
                            .map(|v: Option<&str>| v.unwrap_or("").len())
                            .sum::<usize>()
                    })
                    .unwrap_or(0),
                _ => n * 8,
            }
        })
        .sum()
}
