pub mod compression;
pub mod error;
pub mod filter_pipeline;
pub mod timing;
mod utils;

pub use compression::data::load_csv;
pub use compression::{
    BaseBitImpl, BaseSelectionContext, BitData, BitDataCompressionInfo, BitDataInfo,
    BitDataReconstructionInfo, BitDataSet, BuildBaseTable, BuildBitDataSet, BuildImageBitDataSet,
    BuildSortedBaseTable, CompressedData, CondensedSamples, DEFAULT_ALIGN_ROWS_TO_WORD,
    DecodeDeltaBaseTable, DecompressAnalytics, DecompressFileData, DecompressRandomAccessHandle,
    DeltaBaseTableData, DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, DeviationData,
    DeviationSample, EgdFile, EncodeData, EncodeDataHuffman, EncodeDataOffsetRLE, EncodedData,
    Entropy, EntropyBitScore, EntropyScoredContext, FORMAT_VERSION, FeatureDataType, FeatureSpec,
    FeatureTransform, FloatScalingMode, GenCondensedSamples, IMAGE_FORMAT_VERSION,
    IMAGE_MAGIC_BYTES, IgdFile, ImageColorModel, ImageColorSpace, ImageGroupingTransform,
    ImageReconstructionInfo, LoadEgdFile, LoadIgdFile, LoadTgdFile, MAGIC_BYTES, DEFAULT_DECIMAL_SCALE,
    OpenImage, PixelGrouping, PreEncodeContext, PreprocessOptions, ReconstructDataFrame,
    RleDeviationOffsetData, SaveEgdFile, SaveIgdFile, SaveTgdFile, SelectBases,
    SelectBasesAdaptive, SelectBasesDebug, SelectBasesProfileAllBits, SelectBasesThreshold,
    TGD_FORMAT_VERSION, TGD_MAGIC_BYTES, TgdFile, calculate_entropy, decompress_igd_to_image,
    decompress_tgd_to_csv, load_and_decompress_egd, load_and_decompress_igd,
    load_and_decompress_tgd, reconstruct_feature_value, reconstruct_to_dataframe,
    write_bitdata_as_image, write_cropped_bitdata_as_image,
};
pub use error::EntroGdError;
pub use filter_pipeline::{Chain, Filter, FilterExt};
pub use timing::ScopedTimer;

pub mod prelude {
    pub use crate::{
        BaseBitImpl, BuildBaseTable, BuildBitDataSet, BuildImageBitDataSet, BuildSortedBaseTable,
        DecodeDeltaBaseTable, DecompressAnalytics, DecompressFileData,
        DecompressRandomAccessHandle, DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, EncodeData,
        EncodeDataHuffman, EncodeDataOffsetRLE, Entropy, Filter, FilterExt, FloatScalingMode,
        GenCondensedSamples, ImageColorModel, ImageColorSpace, ImageGroupingTransform, LoadEgdFile,
        LoadIgdFile, LoadTgdFile, OpenImage, PixelGrouping, PreprocessOptions,
        ReconstructDataFrame, SaveEgdFile, SaveIgdFile, SaveTgdFile, SelectBases,
        SelectBasesAdaptive, SelectBasesDebug, SelectBasesProfileAllBits, SelectBasesThreshold,
    };
}

use std::sync::Once;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

static LOGGING_INIT: Once = Once::new();

/// RAII handle for the async logging worker.
///
/// Keep this value alive for as long as you want logging to remain active.
/// Dropping it flushes pending log events.
pub struct LogHandle {
    _guard: WorkerGuard,
}

/// Initialize logging to a file.
///
/// Configuration via environment variables:
/// - `LOG` or `RUST_LOG` (default: `info`)
/// - `ENTRO_GD_LOG_DIR` (default: `logs`)
/// - `ENTRO_GD_LOG_TO_STDERR` (`1`/`true` to mirror `warn+` to stderr)
///
/// Log filename: `gdcompress_YYYYMMDD_HHMMSS.log`.
///
/// Safe to call multiple times. Only the first successful call initializes
/// the global subscriber and returns a [`LogHandle`].
pub fn init_logging() -> Option<LogHandle> {
    let mut log_handle = None;
    LOGGING_INIT.call_once(|| {
        let level_spec = std::env::var("LOG")
            .or_else(|_| std::env::var("RUST_LOG"))
            .unwrap_or_else(|_| "info".to_string());
        let log_dir = std::env::var("ENTRO_GD_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let mirror_stderr = std::env::var("ENTRO_GD_LOG_TO_STDERR")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        if std::fs::create_dir_all(&log_dir).is_err() {
            tracing::warn!(log_dir = %log_dir, "failed to create log directory");
            return;
        }

        let file_name = format!(
            "gdcompress_{}.log",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
        let file_appender = tracing_appender::rolling::never(&log_dir, file_name);
        let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

        let init_result = if mirror_stderr {
            let env_filter =
                EnvFilter::try_new(level_spec.clone()).unwrap_or_else(|_| EnvFilter::new("info"));
            let file_layer = fmt::layer().with_ansi(false).with_writer(file_writer);
            let stderr_layer = fmt::layer().with_ansi(true).with_writer(std::io::stderr);
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(file_layer)
                    .with(stderr_layer),
            )
        } else {
            let env_filter =
                EnvFilter::try_new(level_spec.clone()).unwrap_or_else(|_| EnvFilter::new("info"));
            let file_layer = fmt::layer().with_ansi(false).with_writer(file_writer);
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(file_layer),
            )
        };

        if init_result.is_err() {
            tracing::warn!("failed to initialize tracing subscriber");
            return;
        }

        log_handle = Some(LogHandle { _guard: guard });

        tracing::info!(
            log_dir = %log_dir,
            level = %level_spec,
            mirror_stderr,
            "logging initialized"
        );
    });
    log_handle
}
