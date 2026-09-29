use bitvec::field::BitField;
use bitvec::prelude::*;
use polars::prelude::*;
use std::fmt::{self, Display};
use std::path::Path;
use tracing::debug;

use crate::error::EntroGdError;
use crate::timing::ScopedTimer;

pub const DEFAULT_ALIGN_ROWS_TO_WORD: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelGrouping(pub u32, pub u32);

impl PixelGrouping {
    pub const fn new(width: u32, height: u32) -> Self {
        Self(width, height)
    }

    pub const fn width(self) -> u32 {
        self.0
    }

    pub const fn height(self) -> u32 {
        self.1
    }

    pub const fn total_pixels(self) -> u32 {
        self.0 * self.1
    }
}

impl Default for PixelGrouping {
    fn default() -> Self {
        Self(1, 1)
    }
}

impl Display for PixelGrouping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.0, self.1)
    }
}

pub(crate) fn aligned_stride(chunk_size: usize, pad_rows_to_word: bool) -> usize {
    if pad_rows_to_word {
        chunk_size.next_multiple_of(usize::BITS as usize)
    } else {
        chunk_size
    }
}

// ---------------------------------------------------------------------------
// Feature schema / metadata
// ---------------------------------------------------------------------------

/// Transform metadata used to recover the original value of a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureTransform {
    /// Stored verbatim (or, for floats, as raw IEEE-754 bits).
    None,
    /// Float scaled by `10^decimal_scale`, then biased by a fixed,
    /// data-independent offset (see [`FloatScalingMode::ScaledSignedInt`]).
    ScaledSignedInt { decimal_scale: u8 },
    /// Signed integer, zero-normalized by subtracting `min_value`
    /// (`min_value` may itself be negative).
    OffsetSignedInt { min_value: i64 },
    /// Unsigned integer, zero-normalized by subtracting `min_value`.
    OffsetUnsignedInt { min_value: u64 },
    /// Float scaled by `10^decimal_scale`, then zero-normalized against the
    /// column's actual (scaled) minimum.
    ScaledOffsetSignedInt { decimal_scale: u8, min_value: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDataType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    UInt(u16),
}

impl TryFrom<DataType> for FeatureDataType {
    type Error = EntroGdError;

    fn try_from(data_type: DataType) -> Result<Self, Self::Error> {
        match data_type {
            DataType::Float16 => Ok(FeatureDataType::Float16),
            DataType::Float32 => Ok(FeatureDataType::Float32),
            DataType::Float64 => Ok(FeatureDataType::Float64),
            DataType::Int8 => Ok(FeatureDataType::Int8),
            DataType::Int16 => Ok(FeatureDataType::Int16),
            DataType::Int32 => Ok(FeatureDataType::Int32),
            DataType::Int64 => Ok(FeatureDataType::Int64),
            DataType::UInt8 => Ok(FeatureDataType::UInt8),
            DataType::UInt16 => Ok(FeatureDataType::UInt16),
            DataType::UInt32 => Ok(FeatureDataType::UInt32),
            DataType::UInt64 => Ok(FeatureDataType::UInt64),
            other => Err(EntroGdError::InvalidDataType {
                message: format!("data type {:?} is not yet supported", other),
            }),
        }
    }
}

impl FeatureDataType {
    pub fn bits(&self) -> usize {
        match self {
            FeatureDataType::Float16 => 16,
            FeatureDataType::Float32 => 32,
            FeatureDataType::Float64 => 64,
            FeatureDataType::Int8 => 8,
            FeatureDataType::Int16 => 16,
            FeatureDataType::Int32 => 32,
            FeatureDataType::Int64 => 64,
            FeatureDataType::UInt8 => 8,
            FeatureDataType::UInt16 => 16,
            FeatureDataType::UInt32 => 32,
            FeatureDataType::UInt64 => 64,
            FeatureDataType::UInt128 => 128,
            FeatureDataType::UInt(bits) => *bits as usize,
        }
    }

    /// Smallest unsigned integer type that can hold every value in
    /// `0..=max_value`.
    pub(crate) fn smallest_unsigned_for(max_value: u128) -> FeatureDataType {
        if max_value <= u8::MAX as u128 {
            FeatureDataType::UInt8
        } else if max_value <= u16::MAX as u128 {
            FeatureDataType::UInt16
        } else if max_value <= u32::MAX as u128 {
            FeatureDataType::UInt32
        } else if max_value <= u64::MAX as u128 {
            FeatureDataType::UInt64
        } else {
            FeatureDataType::UInt128
        }
    }
}

/// Per-feature schema entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub data_type: FeatureDataType,
    pub transform: FeatureTransform,
}

impl FeatureSpec {
    pub fn new(data_type: FeatureDataType) -> Self {
        FeatureSpec {
            data_type,
            transform: FeatureTransform::None,
        }
    }
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataCompressionInfo {
    pub features: Vec<FeatureSpec>,
    pub original_size_bits: usize,
    pub n_data_samples: usize,
    pub m_condensed_samples: Option<usize>,
    pub m_condensed_sample_weights: Option<Vec<usize>>,
    feature_offsets: Vec<usize>,
    chunk_size: usize,
    row_stride_bits: usize,
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataInfo {
    pub compression: BitDataCompressionInfo,
    pub reconstruction: BitDataReconstructionInfo,
}

impl BitDataInfo {
    pub fn new(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
    ) -> Result<Self, EntroGdError> {
        let num_features = features.len();
        let column_names: Vec<String> = (0..num_features).map(|i| format!("f{}", i)).collect();
        let original_dtypes = vec![DataType::Float64; num_features];
        Self::new_with_reconstruction_info(
            features,
            original_size_bits,
            BitDataReconstructionInfo::Tabular {
                column_names,
                original_dtypes,
            },
        )
    }

    pub fn new_with_reconstruction_info(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
        reconstruction: BitDataReconstructionInfo,
    ) -> Result<Self, EntroGdError> {
        debug!(
            "Creating BitDataInfo with {} features and original_size_bits={}.",
            features.len(),
            original_size_bits
        );
        if features.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "features is empty".to_string(),
            });
        }

        let mut offsets = Vec::with_capacity(features.len());
        let mut running = 0usize;
        for spec in features.iter() {
            offsets.push(running);
            running += spec.data_type.bits();
        }

        let compression = BitDataCompressionInfo {
            features,
            original_size_bits,
            n_data_samples: original_size_bits / running,
            m_condensed_samples: None,
            m_condensed_sample_weights: None,
            feature_offsets: offsets,
            chunk_size: running,
            row_stride_bits: running,
        };

        let info = BitDataInfo {
            compression,
            reconstruction,
        };

        debug!(
            "BitDataInfo ready: chunk_size={} bits, n_data_samples={}",
            info.chunk_size(),
            info.n_data_samples()
        );

        Ok(info)
    }

    pub fn features(&self) -> &[FeatureSpec] {
        &self.compression.features
    }

    pub fn feature_spec(&self, feature_idx: usize) -> &FeatureSpec {
        &self.compression.features[feature_idx]
    }

    pub fn original_size_bits(&self) -> usize {
        self.compression.original_size_bits
    }

    pub fn n_data_samples(&self) -> usize {
        self.compression.n_data_samples
    }

    pub fn m_condensed_samples(&self) -> Option<usize> {
        self.compression.m_condensed_samples
    }

    pub fn m_condensed_sample_weights(&self) -> Option<&[usize]> {
        self.compression.m_condensed_sample_weights.as_deref()
    }

    pub fn set_condensed_sample_weights(&mut self, weights: Option<Vec<usize>>) {
        self.compression.m_condensed_samples = weights.as_ref().map(Vec::len);
        self.compression.m_condensed_sample_weights = weights;
    }

    pub fn set_tabular_column_info(
        &mut self,
        names: Vec<String>,
        dtypes: Vec<DataType>,
    ) -> Result<(), EntroGdError> {
        let num_features = self.num_features();
        match &mut self.reconstruction {
            BitDataReconstructionInfo::Tabular {
                column_names,
                original_dtypes,
            } => {
                if names.len() != num_features {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "column names count {} does not match feature count {}",
                            names.len(),
                            num_features
                        ),
                    });
                }
                if dtypes.len() != num_features {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "dtype count {} does not match feature count {}",
                            dtypes.len(),
                            num_features
                        ),
                    });
                }
                *column_names = names;
                *original_dtypes = dtypes;
                Ok(())
            }
            BitDataReconstructionInfo::Image(_) => Err(EntroGdError::InvalidMetadata {
                message: "cannot set tabular column info on image reconstruction data".to_string(),
            }),
        }
    }

    pub fn num_features(&self) -> usize {
        self.compression.features.len()
    }

    pub fn chunk_size(&self) -> usize {
        self.compression.chunk_size
    }

    pub fn row_stride(&self) -> usize {
        self.compression.row_stride_bits
    }

    pub fn is_row_padded_to_word(&self) -> bool {
        self.compression.row_stride_bits > self.compression.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.compression.features[feature_idx].data_type.bits()
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.compression.feature_offsets[feature_idx]
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        if bit_pos >= self.compression.chunk_size {
            return None;
        }
        match self.compression.feature_offsets.binary_search(&bit_pos) {
            Ok(idx) => Some(idx),
            Err(0) => None,
            Err(idx) => Some(idx - 1),
        }
    }

    pub fn with_original_size_bits(&self, original_size_bits: usize) -> Self {
        self.with_original_size_bits_and_row_stride(original_size_bits, self.row_stride())
    }

    pub fn with_original_size_bits_and_row_stride(
        &self,
        original_size_bits: usize,
        row_stride_bits: usize,
    ) -> Self {
        let chunk_size = self.compression.chunk_size;
        let row_stride_bits = row_stride_bits.max(chunk_size);
        BitDataInfo {
            compression: BitDataCompressionInfo {
                features: self.compression.features.clone(),
                original_size_bits,
                n_data_samples: original_size_bits / chunk_size,
                m_condensed_samples: self.compression.m_condensed_samples,
                m_condensed_sample_weights: self.compression.m_condensed_sample_weights.clone(),
                feature_offsets: self.compression.feature_offsets.clone(),
                chunk_size,
                row_stride_bits,
            },
            reconstruction: self.reconstruction.clone(),
        }
    }
}

/// Represents the bit-level preprocessed data where each row is a contiguous chunk of bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitData {
    /// The underlying bit storage - all chunks stored contiguously
    pub data: BitVec<usize, Lsb0>,
    /// Logical bits per chunk (what algorithms see)
    pub chunk_size: usize,
    /// Physical bits per chunk including word-alignment padding
    pub stride: usize,
    /// Number of rows/chunks (number of records)
    pub num_rows: usize,
}

impl BitData {
    /// Extend the BitData with additional bits from a BitSlice (used for adding condensed samples)
    pub fn extend_from_bitslice(
        &mut self,
        bits: &BitSlice<usize, Lsb0>,
    ) -> Result<(), EntroGdError> {
        if bits.len() != self.chunk_size {
            return Err(EntroGdError::BitSliceLengthMismatch {
                expected: self.chunk_size,
                actual: bits.len(),
            });
        }
        self.data.extend(bits);
        append_row_padding(&mut self.data, self.stride.saturating_sub(self.chunk_size));
        self.num_rows += 1;
        Ok(())
    }

    /// Get a slice of bits for a specific row/chunk
    #[inline(always)]
    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        let start = row * self.stride;
        let end = start + self.chunk_size;
        &self.data[start..end]
    }

    /// Get a specific bit by linear bit index without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `bit_idx < self.data.len()`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_linear_unchecked(&self, bit_idx: usize) -> bool {
        debug_assert!(bit_idx < self.data.len());
        unsafe { *self.data.get_unchecked(bit_idx) }
    }

    /// Get a specific bit by row and bit position within the chunk
    #[inline(always)]
    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        let idx = row * self.stride + bit_in_chunk;
        self.data[idx]
    }

    /// Get a specific bit by row and bit position within the chunk without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.num_rows` and `bit_in_chunk < self.chunk_size`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_unchecked(&self, row: usize, bit_in_chunk: usize) -> bool {
        debug_assert!(row < self.num_rows);
        debug_assert!(bit_in_chunk < self.chunk_size);
        let idx = row * self.stride + bit_in_chunk;
        unsafe { self.get_bit_linear_unchecked(idx) }
    }

    /// Get raw access to the underlying bit vector
    pub fn raw(&self) -> &BitVec<usize, Lsb0> {
        &self.data
    }

    /// Get the total number of bits stored
    pub fn total_bits(&self) -> usize {
        self.chunk_size * self.num_rows
    }
}

pub(crate) fn append_row_padding(data: &mut BitVec<usize, Lsb0>, padding_bits: usize) {
    if padding_bits > 0 {
        data.resize(data.len() + padding_bits, false);
    }
}

/// Combines BitData with its metadata for algorithms that need both.
#[derive(Debug, Clone)]
pub struct BitDataSet {
    pub data: BitData,
    pub info: BitDataInfo,
}

impl BitDataSet {
    /// Get a slice of bits for a specific feature within a row
    #[inline(always)]
    pub fn get_feature(&self, row: usize, feature: usize) -> &BitSlice<usize, Lsb0> {
        let chunk_start = row * self.data.stride;
        let feat_start = chunk_start + self.info.feature_offset(feature);
        let feat_end = feat_start + self.info.feature_bits(feature);
        &self.data.data[feat_start..feat_end]
    }

    /// Get a specific bit by row, feature, and bit index within the feature
    #[inline(always)]
    pub fn get_bit_by_feature(&self, row: usize, feature: usize, bit: usize) -> bool {
        let idx = row * self.data.stride + self.info.feature_offset(feature) + bit;
        self.data.data[idx]
    }

    /// Get a feature slice without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows` and `feature < self.info.num_features()`.
    #[inline(always)]
    pub(crate) unsafe fn get_feature_unchecked(
        &self,
        row: usize,
        feature: usize,
    ) -> &BitSlice<usize, Lsb0> {
        debug_assert!(row < self.data.num_rows);
        debug_assert!(feature < self.info.num_features());
        let chunk_start = row * self.data.stride;
        let feature_offset = self.info.feature_offset(feature);
        let feature_bits = self.info.feature_bits(feature);
        let feat_start = chunk_start + feature_offset;
        let feat_end = feat_start + feature_bits;
        debug_assert!(feat_end <= self.data.data.len());
        unsafe { self.data.data.get_unchecked(feat_start..feat_end) }
    }

    /// Get a chunk without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows`.
    #[inline(always)]
    pub(crate) unsafe fn get_chunk_unchecked(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        debug_assert!(row < self.data.num_rows);
        let start = row * self.data.stride;
        let end = start + self.data.chunk_size;
        unsafe { self.data.data.get_unchecked(start..end) }
    }

    /// Get a bit without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows` and `bit_in_chunk < self.data.chunk_size`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_unchecked(&self, row: usize, bit_in_chunk: usize) -> bool {
        debug_assert!(row < self.data.num_rows);
        debug_assert!(bit_in_chunk < self.data.chunk_size);
        unsafe { self.data.get_bit_unchecked(row, bit_in_chunk) }
    }

    pub fn num_rows(&self) -> usize {
        self.data.num_rows
    }

    pub fn num_features(&self) -> usize {
        self.info.num_features()
    }

    pub fn chunk_size(&self) -> usize {
        self.data.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.info.feature_bits(feature_idx)
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.info.feature_offset(feature_idx)
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        self.info.feature_index_for_bit(bit_pos)
    }

    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        self.data.get_bit(row, bit_in_chunk)
    }

    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        self.data.get_chunk(row)
    }
}

impl Display for BitDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BitData:")?;
        writeln!(f, "  Number of rows: {}", self.data.num_rows)?;
        writeln!(f, "  Number of features: {}", self.info.num_features())?;
        writeln!(f, "  Chunk size (bits per row): {}", self.data.chunk_size)?;
        writeln!(f, "  Total bits: {}", self.data.total_bits())?;
        write!(f, "Bit  : ")?;
        for bit_pos in 0..self.data.chunk_size {
            write!(f, "{} ", bit_pos)?;
        }
        for row in 0..self.data.num_rows {
            writeln!(f)?;
            write!(f, "Row {}: ", row)?;
            for bit_pos in 0..self.data.chunk_size {
                let bit = self.data.get_bit(row, bit_pos);
                write!(f, "{} ", if bit { '1' } else { '0' })?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Image reconstruction metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGroupingTransform {
    /// Store grouped channel bytes directly.
    Raw = 0,
    /// Store the first byte, then zig-zag encoded residuals relative to it.
    ForFirstPixel = 1,
    /// Store the minimum byte, then zig-zag encoded residuals relative to it.
    ForMin = 2,
}

impl ImageGroupingTransform {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Result<Self, EntroGdError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::ForFirstPixel),
            2 => Ok(Self::ForMin),
            _ => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image grouping transform {} (expected 0, 1, or 2)",
                    value
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorModel {
    /// Store channels as RGB(A).
    Rgb = 0,
    /// Store channels as YCoCg(A), where Co/Cg are biased into byte range.
    YCoCg = 1,
    /// Store channels as reversible YCoCg-R(A), where Co/Cg are wrapped to 8-bit and biased.
    YCoCgR = 2,
}

impl ImageColorModel {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Result<Self, EntroGdError> {
        match value {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::YCoCg),
            2 => Ok(Self::YCoCgR),
            _ => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image color model {} (expected 0, 1, or 2)",
                    value
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageReconstructionInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub color_model: ImageColorModel,
    pub pixel_grouping: PixelGrouping,
    pub grouping_transform: ImageGroupingTransform,
    /// 0 = sRGB + linear alpha, 1 = all linear
    pub colorspace: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitDataReconstructionInfo {
    Tabular {
        column_names: Vec<String>,
        original_dtypes: Vec<DataType>,
    },
    Image(ImageReconstructionInfo),
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct the original value of a feature from its bit-packed
/// representation by reversing the stored [`FeatureTransform`].
pub fn reconstruct_feature_value(bits: &BitSlice<usize, Lsb0>, spec: &FeatureSpec) -> f64 {
    let raw: u64 = bits.load_le::<u64>();

    match spec.transform {
        FeatureTransform::None => match spec.data_type {
            FeatureDataType::Float16 => f64::from(f32::from_bits((raw as u32) << 16)),
            FeatureDataType::Float32 => f64::from(f32::from_bits(raw as u32)),
            FeatureDataType::Float64 => f64::from_bits(raw),
            FeatureDataType::UInt128 => {
                let raw128: u128 = bits.load_le::<u128>();
                raw128 as f64
            }
            _ => raw as f64,
        },
        FeatureTransform::ScaledSignedInt { decimal_scale } => {
            let signed = (raw as i128).wrapping_add(i64::MIN as i128);
            signed as f64 / 10f64.powi(decimal_scale as i32)
        }
        FeatureTransform::OffsetSignedInt { min_value } => {
            (raw as i64).wrapping_add(min_value) as f64
        }
        FeatureTransform::OffsetUnsignedInt { min_value } => (raw + min_value) as f64,
        FeatureTransform::ScaledOffsetSignedInt {
            decimal_scale,
            min_value,
        } => {
            let signed = (raw as i64).wrapping_add(min_value) as f64;
            signed / 10f64.powi(decimal_scale as i32)
        }
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

pub fn load_csv(
    path: impl AsRef<Path>,
    has_header: bool,
    null_strategy: Option<FillNullStrategy>,
) -> Result<DataFrame, EntroGdError> {
    let _timer = ScopedTimer::info(format!("Loading dataset from CSV: {:?}", path.as_ref()));

    let mut df = CsvReadOptions::default()
        .with_has_header(has_header)
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
        .finish()?;

    if let Some(strategy) = null_strategy {
        df = df.fill_null(strategy)?;
    }

    Ok(df)
}
