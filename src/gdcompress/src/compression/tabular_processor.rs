use bitvec::prelude::*;
use polars::prelude::*;
use std::path::{Path, PathBuf};
use tracing::{debug, info, trace};

use super::data::{
    self, BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, DEFAULT_ALIGN_ROWS_TO_WORD,
    FeatureDataType, FeatureSpec, FeatureTransform, reconstruct_feature_value,
};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

pub const DEFAULT_DECIMAL_SCALE: u8 = 9;

// ---------------------------------------------------------------------------
// Tabular ingestion
// ---------------------------------------------------------------------------

/// Where a tabular [`DataFrame`] comes from. Every variant just needs to be
/// able to produce a `DataFrame`; everything downstream (transforms, bit
/// packing) only ever operates on that `DataFrame`, so adding a new source
/// format (Parquet, JSON, Arrow IPC, ...) later is purely additive.
#[derive(Debug, Clone)]
pub enum TabularSource {
    Csv { path: PathBuf },
}

impl TabularSource {
    pub fn csv(path: impl Into<PathBuf>) -> Self {
        TabularSource::Csv { path: path.into() }
    }

    pub fn load(&self) -> Result<DataFrame, EntroGdError> {
        match self {
            TabularSource::Csv { path } => load_csv(path),
        }
    }
}

/// Load a CSV file into a Polars [`DataFrame`] using Polars' own reader
/// rather than a hand-rolled parser, so column typing, quoting, and encoding
/// edge cases are handled for us.
pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<DataFrame, EntroGdError> {
    let path = path.as_ref();
    trace!("Loading CSV dataset from {}", path.display());

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))
        .map_err(|e| EntroGdError::InvalidDataType {
            message: format!("failed to open CSV '{}': {}", path.display(), e),
        })?
        .finish()
        .map_err(|e| EntroGdError::InvalidDataType {
            message: format!("failed to parse CSV '{}': {}", path.display(), e),
        })?;

    info!(
        "Loaded CSV '{}': {} row(s) x {} column(s)",
        path.display(),
        df.height(),
        df.width()
    );

    Ok(df)
}

// ---------------------------------------------------------------------------
// Preprocessing options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatScalingMode {
    /// Store the raw IEEE-754 bits of the original float, unmodified.
    Disabled,
    /// Scale to an integer, then bias by a fixed, data-independent offset
    /// (derived from the integer type's range) instead of scanning the
    /// column for its true minimum. Faster, but does not shrink the bit
    /// width beyond the scaled type's natural size.
    ScaledSignedInt,
    /// Scale to an integer, then zero-normalize using the column's actual
    /// minimum so the value range fits the smallest possible unsigned
    /// integer. This is the default: best compression, small extra scan.
    ScaledOffsetSignedInt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessOptions {
    /// How (or whether) floating point columns are converted to integers.
    pub float_scaling: FloatScalingMode,
    /// Number of decimal places to preserve when scaling floats to
    /// integers, e.g. `decimal_scale = 2` multiplies by 100 before
    /// rounding. This is a fixed, user-supplied precision rather than
    /// something searched for per-column, which keeps preprocessing fast.
    pub decimal_scale: u8,
    /// Whether integer columns (signed or unsigned) are zero-normalized
    /// against their actual observed minimum. When disabled, signed
    /// columns still have to become unsigned to be bit-packed, but do so
    /// via a fixed type-range bias instead of a data scan.
    pub integer_zero_normalization: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        PreprocessOptions {
            float_scaling: FloatScalingMode::ScaledOffsetSignedInt,
            decimal_scale: DEFAULT_DECIMAL_SCALE,
            integer_zero_normalization: true,
        }
    }
}

/// Filter for building a [`BitDataSet`] from a [`DataFrame`].
#[derive(Debug, Clone, Copy)]
pub struct BuildBitDataSet {
    pub options: PreprocessOptions,
    pub pad_rows_to_word: bool,
}

impl Default for BuildBitDataSet {
    fn default() -> Self {
        Self {
            options: PreprocessOptions::default(),
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
    }
}

impl Filter for BuildBitDataSet {
    type Input = DataFrame;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building BitDataSet");
        BitDataSet::from_dataframe(input, self.options, self.pad_rows_to_word)
    }
}

// ---------------------------------------------------------------------------
// BitDataSet construction from tabular data
// ---------------------------------------------------------------------------

impl BitDataSet {
    /// Build a [`BitDataSet`] directly from a [`TabularSource`] (loads then transforms).
    pub fn from_source(
        source: &TabularSource,
        options: PreprocessOptions,
        pad_rows_to_word: bool,
    ) -> Result<Self, EntroGdError> {
        let dataframe = source.load()?;
        Self::from_dataframe(dataframe, options, pad_rows_to_word)
    }

    /// Create a `BitDataSet` from a `DataFrame`:
    ///   1. Transform every column (zero-normalize integers, scale + zero-normalize
    ///      floats) and pick the smallest unsigned integer type that fits it.
    ///   2. Preallocate the destination bit buffer to its final size
    ///      (`row_stride_bits * num_rows`) up front.
    ///   3. Fill the buffer one column (feature) at a time, which keeps memory
    ///      access to the source data sequential even though writes into the
    ///      row-major bit buffer are strided.
    pub fn from_dataframe(
        dataframe: DataFrame,
        options: PreprocessOptions,
        pad_rows_to_word: bool,
    ) -> Result<Self, EntroGdError> {
        let num_rows = dataframe.height();
        if num_rows == 0 {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "cannot build a BitDataSet from an empty dataframe".to_string(),
            });
        }

        let columns = dataframe.columns();
        if columns.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "dataframe has no columns".to_string(),
            });
        }

        info!(
            "Preprocessing {} column(s) x {} row(s) into a BitDataSet",
            columns.len(),
            num_rows
        );

        let transformed: Vec<TransformedColumn> = columns
            .iter()
            .map(|col| {
                let series = col.as_materialized_series();
                transform_column(&series, &options)
            })
            .collect::<Result<_, _>>()?;

        let features: Vec<FeatureSpec> = transformed.iter().map(|t| t.spec.clone()).collect();
        let chunk_size: usize = features.iter().map(|f| f.data_type.bits()).sum();
        let stride = data::aligned_stride(chunk_size, pad_rows_to_word);
        let original_size_bits = num_rows * chunk_size;

        let column_names: Vec<String> = columns.iter().map(|c| c.name().to_string()).collect();
        let original_dtypes: Vec<DataType> = columns.iter().map(|c| c.dtype().clone()).collect();

        let mut info = BitDataInfo::new_with_reconstruction_info(
            features,
            original_size_bits,
            BitDataReconstructionInfo::Tabular {
                column_names,
                original_dtypes,
            },
        )?;
        if stride != chunk_size {
            info = info.with_original_size_bits_and_row_stride(original_size_bits, stride);
        }

        debug!(
            "BitDataSet layout: chunk_size={} bits, stride={} bits, num_rows={}",
            chunk_size, stride, num_rows
        );

        // Preallocate the whole buffer up front rather than growing it row by row.
        let mut bits: BitVec<usize, Lsb0> = BitVec::repeat(false, stride * num_rows);

        // Fill column-by-column: for each feature, walk its already-transformed
        // UInt64 Series and store each value into its row slot.
        for (feature_idx, column) in transformed.iter().enumerate() {
            let bit_width = info.feature_bits(feature_idx);
            let base_offset = info.feature_offset(feature_idx);
            let ca = column
                .series
                .u64()
                .map_err(|e| EntroGdError::InvalidDataType {
                    message: format!(
                        "internal error: transformed column '{}' is not UInt64: {}",
                        column.series.name(),
                        e
                    ),
                })?;
            for (row, value) in ca.into_no_null_iter().enumerate() {
                let start = row * stride + base_offset;
                let end = start + bit_width;
                bits[start..end].store_le::<u64>(value);
            }
        }

        let data = BitData {
            data: bits,
            chunk_size,
            stride,
            num_rows,
        };

        Ok(BitDataSet { data, info })
    }
}

// ---------------------------------------------------------------------------
// Column transforms
// ---------------------------------------------------------------------------

/// A column after transformation: its final schema entry, plus the
/// transformed `Series` (UInt64) holding the packable value for every row.
struct TransformedColumn {
    spec: FeatureSpec,
    series: Series,
}

fn apply_transform(s: &Series, transform: &FeatureTransform) -> PolarsResult<Series> {
    match transform {
        FeatureTransform::None => match s.dtype() {
            DataType::Float64 => {
                let ca = s.f64()?;
                let out: UInt64Chunked =
                    ca.apply_nonnull_values_generic(DataType::UInt64, |v| v.to_bits());
                Ok(out.into_series())
            }
            DataType::Float32 => {
                let ca = s.f32()?;
                let out: UInt64Chunked =
                    ca.apply_nonnull_values_generic(DataType::UInt64, |v| v.to_bits() as u64);
                Ok(out.into_series())
            }
            _ => s.cast(&DataType::UInt64),
        },

        FeatureTransform::ScaledSignedInt { decimal_scale } => {
            let factor = 10f64.powi(*decimal_scale as i32);
            let ca = s.f64()?;
            let out: UInt64Chunked = ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                let scaled = (v * factor).round() as i64;
                (scaled as u64).wrapping_sub(i64::MIN as u64)
            });
            Ok(out.into_series())
        }

        FeatureTransform::OffsetSignedInt { min_value } => {
            let ca = s.i64()?;
            let out: UInt64Chunked = ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                (v as u64).wrapping_sub(*min_value as u64)
            });
            Ok(out.into_series())
        }

        FeatureTransform::OffsetUnsignedInt { min_value } => {
            let ca = s.u64()?;
            let out: UInt64Chunked =
                ca.apply_nonnull_values_generic(DataType::UInt64, |v| v - min_value);
            Ok(out.into_series())
        }

        FeatureTransform::ScaledOffsetSignedInt {
            decimal_scale,
            min_value,
        } => {
            let factor = 10f64.powi(*decimal_scale as i32);
            let ca = s.f64()?;
            let out: UInt64Chunked = ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                let scaled = (v * factor).round() as i64;
                (scaled as u64).wrapping_sub(*min_value as u64)
            });
            Ok(out.into_series())
        }
    }
}

fn transform_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    if series.null_count() > 0 {
        return Err(EntroGdError::InvalidDataType {
            message: format!(
                "column '{}' contains null values, which are not yet supported",
                series.name()
            ),
        });
    }

    match series.dtype() {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            transform_float_column(series, options)
        }
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            transform_signed_column(series, options)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            transform_unsigned_column(series, options)
        }
        other => Err(EntroGdError::InvalidDataType {
            message: format!(
                "column '{}' has unsupported dtype {:?}",
                series.name(),
                other
            ),
        }),
    }
}

fn cast_err(column: &str, e: PolarsError) -> EntroGdError {
    EntroGdError::InvalidDataType {
        message: format!("failed to cast column '{}': {}", column, e),
    }
}

fn empty_column_err(column: &str) -> EntroGdError {
    EntroGdError::InvalidFeatureSpec {
        message: format!("column '{}' has no rows to derive a range from", column),
    }
}

/// Fixed, data-independent bias for mapping a signed integer type to
/// unsigned without scanning the data (used when zero-normalization is
/// disabled). This is just the type's natural minimum, i.e. the standard
/// "flip the sign bit" trick.
fn fixed_signed_bias(dtype: &DataType) -> i64 {
    match dtype {
        DataType::Int8 => i8::MIN as i64,
        DataType::Int16 => i16::MIN as i64,
        DataType::Int32 => i32::MIN as i64,
        DataType::Int64 => i64::MIN,
        _ => 0,
    }
}

fn transform_signed_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();
    let casted = series
        .cast(&DataType::Int64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.i64().map_err(|e| cast_err(&name, e))?;

    let min_value = if options.integer_zero_normalization {
        ca.min().ok_or_else(|| empty_column_err(&name))?
    } else {
        fixed_signed_bias(series.dtype())
    };

    let transform = FeatureTransform::OffsetSignedInt { min_value };
    let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
    let range = out
        .u64()
        .map_err(|e| cast_err(&name, e))?
        .max()
        .unwrap_or(0);

    Ok(TransformedColumn {
        spec: FeatureSpec {
            data_type: FeatureDataType::smallest_unsigned_for(range as u128),
            transform,
        },
        series: out,
    })
}

fn transform_unsigned_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();
    let casted = series
        .cast(&DataType::UInt64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.u64().map_err(|e| cast_err(&name, e))?;

    if !options.integer_zero_normalization {
        let max_value = ca.max().ok_or_else(|| empty_column_err(&name))?;
        return Ok(TransformedColumn {
            spec: FeatureSpec {
                data_type: FeatureDataType::smallest_unsigned_for(max_value as u128),
                transform: FeatureTransform::None,
            },
            series: casted,
        });
    }

    let min_value = ca.min().ok_or_else(|| empty_column_err(&name))?;
    let transform = FeatureTransform::OffsetUnsignedInt { min_value };
    let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
    let range = out
        .u64()
        .map_err(|e| cast_err(&name, e))?
        .max()
        .unwrap_or(0);

    Ok(TransformedColumn {
        spec: FeatureSpec {
            data_type: FeatureDataType::smallest_unsigned_for(range as u128),
            transform,
        },
        series: out,
    })
}

fn transform_float_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();

    if options.float_scaling == FloatScalingMode::Disabled {
        let data_type: FeatureDataType = series.dtype().clone().try_into()?;
        let transform = FeatureTransform::None;
        let out = apply_transform(series, &transform).map_err(|e| cast_err(&name, e))?;
        return Ok(TransformedColumn {
            spec: FeatureSpec {
                data_type,
                transform,
            },
            series: out,
        });
    }

    let casted = series
        .cast(&DataType::Float64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.f64().map_err(|e| cast_err(&name, e))?;

    if options.decimal_scale > DEFAULT_DECIMAL_SCALE {
        tracing::warn!(
            "decimal_scale ({}) exceeds {}; you may get overflow errors for large float values",
            options.decimal_scale,
            DEFAULT_DECIMAL_SCALE,
        );
    }
    let decimal_scale = options.decimal_scale;
    let multiplier = 10f64.powi(decimal_scale as i32);

    let scaled_min = ca.into_no_null_iter().try_fold(i64::MAX, |min_acc, v| {
        if !v.is_finite() {
            return Err(EntroGdError::InvalidDataType {
                message: format!(
                    "column '{}' contains a non-finite value ({}) that cannot be scaled",
                    name, v
                ),
            });
        }
        let s = v * multiplier;
        if s < i64::MIN as f64 || s > i64::MAX as f64 {
            return Err(EntroGdError::InvalidDataType {
                message: format!(
                    "column '{}' overflows i64 once scaled by 10^{}",
                    name, decimal_scale
                ),
            });
        }
        Ok(min_acc.min(s.round() as i64))
    })?;

    match options.float_scaling {
        FloatScalingMode::Disabled => unreachable!(),
        FloatScalingMode::ScaledSignedInt => {
            let transform = FeatureTransform::ScaledSignedInt { decimal_scale };
            let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
            let range = out
                .u64()
                .map_err(|e| cast_err(&name, e))?
                .max()
                .unwrap_or(0);
            Ok(TransformedColumn {
                spec: FeatureSpec {
                    data_type: FeatureDataType::smallest_unsigned_for(range as u128),
                    transform,
                },
                series: out,
            })
        }
        FloatScalingMode::ScaledOffsetSignedInt => {
            let transform = FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value: scaled_min,
            };
            let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
            let range = out
                .u64()
                .map_err(|e| cast_err(&name, e))?
                .max()
                .unwrap_or(0);
            Ok(TransformedColumn {
                spec: FeatureSpec {
                    data_type: FeatureDataType::smallest_unsigned_for(range as u128),
                    transform,
                },
                series: out,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct a [`BitDataSet`] into a [`DataFrame`] by reversing
/// per-feature transforms. Column names and original data types are
/// read from the dataset's reconstruction metadata.
pub fn reconstruct_to_dataframe(dataset: &BitDataSet) -> Result<DataFrame, EntroGdError> {
    let (column_names, original_dtypes) = match &dataset.info.reconstruction {
        BitDataReconstructionInfo::Tabular {
            column_names,
            original_dtypes,
        } => (column_names.clone(), original_dtypes.clone()),
        BitDataReconstructionInfo::Image(_) => {
            return Err(EntroGdError::InvalidMetadata {
                message: "cannot reconstruct tabular DataFrame from image reconstruction data"
                    .to_string(),
            });
        }
    };

    if column_names.len() != dataset.num_features() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "column names count {} does not match feature count {}",
                column_names.len(),
                dataset.num_features()
            ),
        });
    }

    let num_rows = dataset.num_rows();
    let num_features = dataset.num_features();

    let mut columns = Vec::with_capacity(num_features);
    for feature_idx in 0..num_features {
        let spec = dataset.info.feature_spec(feature_idx);
        let values: Vec<f64> = (0..num_rows)
            .map(|row| {
                let bits = unsafe { dataset.get_feature_unchecked(row, feature_idx) };
                reconstruct_feature_value(bits, spec)
            })
            .collect();

        let series = Series::new(column_names[feature_idx].clone().into(), values);

        let casted = if original_dtypes[feature_idx] != DataType::Float64 {
            series.cast(&original_dtypes[feature_idx]).map_err(|e| {
                EntroGdError::InvalidDataType {
                    message: format!(
                        "failed to cast column '{}' to {:?}: {}",
                        column_names[feature_idx], original_dtypes[feature_idx], e
                    ),
                }
            })?
        } else {
            series
        };

        columns.push(casted.into());
    }

    DataFrame::new(num_rows, columns).map_err(|e| EntroGdError::InvalidDataType {
        message: format!("failed to build DataFrame: {}", e),
    })
}

/// [`Filter`] that reconstructs a [`BitDataSet`] into a [`DataFrame`].
#[derive(Debug, Clone, Copy)]
pub struct ReconstructDataFrame;

impl Filter for ReconstructDataFrame {
    type Input = BitDataSet;
    type Output = DataFrame;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Reconstructing DataFrame from BitDataSet");
        reconstruct_to_dataframe(&input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::data::{FeatureDataType, FeatureTransform};

    fn load_le(slice: &BitSlice<usize, Lsb0>) -> u64 {
        slice.load_le::<u64>()
    }

    #[test]
    fn smallest_unsigned_for_picks_tightest_type() {
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(0),
            FeatureDataType::UInt8
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(255),
            FeatureDataType::UInt8
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(256),
            FeatureDataType::UInt16
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u16::MAX as u128),
            FeatureDataType::UInt16
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u16::MAX as u128 + 1),
            FeatureDataType::UInt32
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u32::MAX as u128),
            FeatureDataType::UInt32
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u32::MAX as u128 + 1),
            FeatureDataType::UInt64
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u64::MAX as u128),
            FeatureDataType::UInt64
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u64::MAX as u128 + 1),
            FeatureDataType::UInt128
        );
    }

    #[test]
    fn signed_column_zero_normalizes_to_smallest_uint() {
        let s = Series::new("temperature".into(), &[-10i32, 0, 5, 20]);
        let options = PreprocessOptions::default();
        let transformed = transform_column(&s, &options).unwrap();

        assert_eq!(transformed.spec.data_type, FeatureDataType::UInt8);
        assert_eq!(
            transformed.spec.transform,
            FeatureTransform::OffsetSignedInt { min_value: -10 }
        );
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![0, 10, 15, 30]);
    }

    #[test]
    fn unsigned_column_zero_normalizes() {
        let s = Series::new("count".into(), &[100u32, 105, 110]);
        let options = PreprocessOptions::default();
        let transformed = transform_column(&s, &options).unwrap();

        assert_eq!(transformed.spec.data_type, FeatureDataType::UInt8);
        assert_eq!(
            transformed.spec.transform,
            FeatureTransform::OffsetUnsignedInt { min_value: 100 }
        );
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![0, 5, 10]);
    }

    #[test]
    fn float_column_scales_and_offsets() {
        let s = Series::new("price".into(), &[1.50f64, 2.25, 0.75]);
        let mut options = PreprocessOptions::default();
        options.decimal_scale = 2;
        let transformed = transform_column(&s, &options).unwrap();

        match transformed.spec.transform {
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                assert_eq!(decimal_scale, 2);
                assert_eq!(min_value, 75);
            }
            other => panic!("unexpected transform: {:?}", other),
        }
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![75, 150, 0]);
    }

    #[test]
    fn from_dataframe_round_trips_values_through_bit_storage() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), &[-5i32, 0, 5]).into(),
                Series::new("b".into(), &[10i32, 20, 30]).into(),
            ],
        )
        .unwrap();

        let dataset = BitDataSet::from_dataframe(df, PreprocessOptions::default(), false).unwrap();
        assert_eq!(dataset.num_rows(), 3);
        assert_eq!(dataset.num_features(), 2);

        // Column "a": min -5 -> offsets 0, 5, 10
        assert_eq!(load_le(dataset.get_feature(0, 0)), 0);
        assert_eq!(load_le(dataset.get_feature(1, 0)), 5);
        assert_eq!(load_le(dataset.get_feature(2, 0)), 10);

        // Column "b": min 10 -> offsets 0, 10, 20
        assert_eq!(load_le(dataset.get_feature(0, 1)), 0);
        assert_eq!(load_le(dataset.get_feature(1, 1)), 10);
        assert_eq!(load_le(dataset.get_feature(2, 1)), 20);
    }
}
