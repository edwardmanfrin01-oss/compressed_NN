use super::rle::RLE_LONG_MAX;
use bitvec::prelude::*;
use rayon::prelude::*;

use crate::compression::base_table::{BaseBitLayoutState, BaseLayoutInfo, PreEncodeContext};
use crate::compression::data::{BitDataInfo, BitDataReconstructionInfo};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;

const RLE_MAX_CONTROL_VALUE: usize = RLE_LONG_MAX as usize;
const RLE_MAX_RUN_LEN: usize = RLE_MAX_CONTROL_VALUE + 1;

/// Represents the compressed output.
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// The encoded data stream.
    pub encoded_data: EncodedData,
    /// The weights for the condensed samples (if used).
    // Should be stored as a bitstream of length m * l_w (log_2(n).ceil() bits per weight).
    pub condensed_sample_weights: Option<Vec<usize>>,
    /// Base table mapping base patterns to their frequencies or encodings.
    pub base_table: BaseTable,
    /// Base-table bit layout metadata.
    pub layout: BaseLayoutInfo,
    /// Column indices into `base_table` row bit-vectors (variable part), ordered
    /// by ascending unweighted entropy as used by `BuildSortedBaseTable`.
    pub entropy_sorted_column_order: Option<Vec<usize>>,
    /// Metadata for decompression (column count, base bits used, etc.).
    pub metadata: BitDataInfo,
}

#[derive(Debug, Clone)]
pub struct CondensedSamples {
    pub samples: Vec<BitVec<usize, Lsb0>>,
    pub weights: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DeviationSample {
    pub deviation: BitVec<usize, Lsb0>,
    pub id: BitVec<usize, Lsb0>,
}

pub(crate) struct DeviationSampleRef<'a> {
    pub deviation: &'a BitSlice<usize, Lsb0>,
    pub id: &'a BitSlice<usize, Lsb0>,
}

#[derive(Debug, Clone)]
pub struct DeviationData {
    pub(super) encoded_bit_stream: BitVec<usize, Lsb0>,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct RleDeviationOffsetData {
    pub(super) symbol_bit_stream: BitVec<usize, Lsb0>,
    pub(super) rm_values: Vec<(u8, u8)>,
    pub(super) rm_control_stream: BitVec<usize, Lsb0>,
    pub(super) row_offset_stream: BitVec<usize, Lsb0>,
    pub(super) row_offsets: Vec<(u32, u32)>,
    pub(super) original_num_samples: usize,
    pub(super) row_width: usize,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct HuffmanDeviationData {
    pub(super) pixel_bit_stream: BitVec<usize, Lsb0>,
    pub(super) raw_deviation_bit_stream: BitVec<usize, Lsb0>,
    pub(super) canonical_symbols: Vec<u64>,
    pub(super) canonical_code_lengths: Vec<u8>,
    pub(super) row_offsets: Vec<u32>,
    pub(super) num_samples: usize,
    pub(super) original_num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
    pub(super) row_width: usize,
    pub(super) max_code_length: u8,
    pub(super) min_code_by_len: Vec<u32>,
    pub(super) max_code_by_len: Vec<u32>,
    pub(super) first_symbol_index_by_len: Vec<usize>,
    pub(super) decode_symbols: Vec<u64>,
    pub(super) original_stream_end_offset_bits: usize,
}

#[derive(Debug, Clone)]
pub enum EncodedData {
    Normal(DeviationData),
    RleOffset(RleDeviationOffsetData),
    Huffman(HuffmanDeviationData),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseTable {
    Raw(Vec<(BitVec<usize, Lsb0>, usize)>),
    Delta(DeltaBaseTableData),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaBaseTableData {
    pub num_bases: usize,
    pub first_sort_key: BitVec<usize, Lsb0>,
    pub delta_bit_stream: BitVec<usize, Lsb0>,
    pub delta_count: usize,
    pub sort_column_order: Vec<usize>,
    /// Codec tag: 1 for unary prefix, 2 for fixed prefix
    pub codec_id: u8,
}

impl BaseTable {
    pub fn as_raw(&self) -> &[(BitVec<usize, Lsb0>, usize)] {
        match self {
            BaseTable::Raw(table) => table.as_slice(),
            BaseTable::Delta(_) => {
                panic!("DecodeDeltaBaseTable must be applied before calling as_raw()")
            }
        }
    }

    pub fn as_raw_mut(&mut self) -> &mut Vec<(BitVec<usize, Lsb0>, usize)> {
        match self {
            BaseTable::Raw(table) => table,
            BaseTable::Delta(_) => {
                panic!("DecodeDeltaBaseTable must be applied before calling as_raw_mut()")
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BaseTable::Raw(table) => table.len(),
            BaseTable::Delta(delta) => delta.num_bases,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            BaseTable::Raw(table) => table.is_empty(),
            BaseTable::Delta(delta) => delta.num_bases == 0,
        }
    }
}

pub(crate) fn huffman_row_layout(
    metadata: &BitDataInfo,
) -> Result<(usize, usize, usize), EntroGdError> {
    let chunk_size = metadata.chunk_size();
    if chunk_size == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "chunk_size is 0".to_string(),
        });
    }

    let original_num_samples = metadata.original_size_bits() / chunk_size;
    match metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => {
            let grouped_width = info.pixel_grouping.width() as usize;
            let grouped_height = info.pixel_grouping.height() as usize;
            if grouped_width == 0 || grouped_height == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "image pixel_grouping width and height must be > 0".to_string(),
                });
            }
            let row_count = (info.height as usize).div_ceil(grouped_height);
            let row_width = (info.width as usize).div_ceil(grouped_width);
            if row_count == 0 && original_num_samples == 0 {
                return Ok((0, 0, 0));
            }
            if row_count == 0 || row_width == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid image row layout width={} height={} pixel_grouping={}x{}",
                        info.width,
                        info.height,
                        info.pixel_grouping.width(),
                        info.pixel_grouping.height()
                    ),
                });
            }
            let expected_samples =
                row_count
                    .checked_mul(row_width)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "image row layout overflows sample count".to_string(),
                    })?;
            if expected_samples != original_num_samples {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "image row layout mismatch: ceil(height / group_height) * ceil(width / group_width) = {}, original samples = {}",
                        expected_samples, original_num_samples
                    ),
                });
            }
            Ok((original_num_samples, row_count, row_width))
        }
        BitDataReconstructionInfo::Tabular { .. } => {
            const TABULAR_OFFSET_WIDTH: usize = 1024;
            let row_count = original_num_samples.div_ceil(TABULAR_OFFSET_WIDTH);
            Ok((original_num_samples, row_count, TABULAR_OFFSET_WIDTH))
        }
    }
}

impl CompressedData {
    pub fn new(encoded_data: EncodedData, metadata: BitDataInfo) -> Self {
        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table: BaseTable::Raw(Vec::new()),
            layout: BaseLayoutInfo::from_bit_states(vec![
                BaseBitLayoutState::Deviation;
                metadata.chunk_size()
            ]),
            entropy_sorted_column_order: None,
            metadata,
        }
    }
}

impl DeviationData {
    pub fn new(
        encoded_bit_stream: BitVec<usize, Lsb0>,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Self {
        DeviationData {
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        }
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }
        let start_bit = sample_idx * (self.num_deviation_bits + self.num_id_bits);
        let end_bit = start_bit + self.num_deviation_bits + self.num_id_bits;
        debug_assert!(end_bit <= self.encoded_bit_stream.len());
        Some(DeviationSample {
            deviation: unsafe {
                self.encoded_bit_stream
                    .get_unchecked(start_bit..start_bit + self.num_deviation_bits)
            }
            .to_bitvec(),
            id: unsafe {
                self.encoded_bit_stream
                    .get_unchecked(start_bit + self.num_deviation_bits..end_bit)
            }
            .to_bitvec(),
        })
    }

    pub fn get_encoded_size(&self) -> usize {
        self.encoded_bit_stream.len()
    }

    pub fn encoded_bit_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.encoded_bit_stream
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        self.num_deviation_bits
    }

    pub fn get_num_id_bits(&self) -> usize {
        self.num_id_bits
    }

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let sample_width = self.num_deviation_bits + self.num_id_bits;
        let capped_limit = limit.min(self.num_samples);

        for sample_idx in 0..capped_limit {
            let start_bit = sample_idx * sample_width;
            let end_bit = start_bit + sample_width;
            debug_assert!(end_bit <= self.encoded_bit_stream.len());
            f(DeviationSampleRef {
                deviation: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit..start_bit + self.num_deviation_bits)
                },
                id: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit + self.num_deviation_bits..end_bit)
                },
            })?;
        }

        Ok(())
    }

    pub(crate) fn for_each_sample_at_sorted_indices(
        &self,
        sorted_indices: &[usize],
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let sample_width = self.num_deviation_bits + self.num_id_bits;
        for &sample_idx in sorted_indices {
            if sample_idx >= self.num_samples {
                return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
            }
            let start_bit = sample_idx * sample_width;
            let end_bit = start_bit + sample_width;
            debug_assert!(end_bit <= self.encoded_bit_stream.len());
            f(DeviationSampleRef {
                deviation: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit..start_bit + self.num_deviation_bits)
                },
                id: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit + self.num_deviation_bits..end_bit)
                },
            })?;
        }
        Ok(())
    }

    pub(crate) fn for_each_sample_range(
        &self,
        start: usize,
        count: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let sample_width = self.num_deviation_bits + self.num_id_bits;
        let end = (start + count).min(self.num_samples);
        for sample_idx in start..end {
            let bit_start = sample_idx * sample_width;
            let bit_end = bit_start + sample_width;
            debug_assert!(bit_end <= self.encoded_bit_stream.len());
            f(DeviationSampleRef {
                deviation: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(bit_start..bit_start + self.num_deviation_bits)
                },
                id: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(bit_start + self.num_deviation_bits..bit_end)
                },
            })?;
        }
        Ok(())
    }
}

impl EncodedData {
    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        match self {
            EncodedData::Normal(data) => data.get_sample(sample_idx),
            EncodedData::RleOffset(data) => data.get_sample(sample_idx),
            EncodedData::Huffman(data) => data.get_sample(sample_idx),
        }
    }

    pub fn get_encoded_size(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_encoded_size(),
            EncodedData::RleOffset(data) => data.get_encoded_size(),
            EncodedData::Huffman(data) => data.get_encoded_size(),
        }
    }

    pub fn encoded_bit_stream(&self) -> &BitVec<usize, Lsb0> {
        match self {
            EncodedData::Normal(data) => data.encoded_bit_stream(),
            EncodedData::RleOffset(data) => data.symbol_bit_stream(),
            EncodedData::Huffman(data) => data.pixel_bit_stream(),
        }
    }

    pub fn get_num_samples(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_samples(),

            EncodedData::RleOffset(data) => data.get_num_samples(),
            EncodedData::Huffman(data) => data.get_num_samples(),
        }
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_deviation_bits(),
            EncodedData::RleOffset(data) => data.get_num_deviation_bits(),
            EncodedData::Huffman(data) => data.get_num_deviation_bits(),
        }
    }

    pub fn get_num_id_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_id_bits(),
            EncodedData::RleOffset(data) => data.get_num_id_bits(),
            EncodedData::Huffman(data) => data.get_num_id_bits(),
        }
    }

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        match self {
            EncodedData::Normal(data) => data.for_each_sample_n(limit, f),
            EncodedData::RleOffset(data) => data.for_each_sample_n(limit, f),
            EncodedData::Huffman(data) => data.for_each_sample_n(limit, f),
        }
    }

    pub(crate) fn for_each_sample_at_sorted_indices(
        &self,
        sorted_indices: &[usize],
        f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        match self {
            EncodedData::Normal(data) => data.for_each_sample_at_sorted_indices(sorted_indices, f),
            EncodedData::RleOffset(data) => {
                data.for_each_sample_at_sorted_indices(sorted_indices, f)
            }
            EncodedData::Huffman(data) => data.for_each_sample_at_sorted_indices(sorted_indices, f),
        }
    }

    /// Decode a contiguous range of samples `[start, start + count)` in a single forward pass.
    /// Not supported for `Rle` (no stored row offsets) — callers must use `for_each_sample_n` there.
    pub(crate) fn for_each_sample_range(
        &self,
        start: usize,
        count: usize,
        f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        match self {
            EncodedData::Normal(data) => data.for_each_sample_range(start, count, f),
            EncodedData::RleOffset(data) => data.for_each_sample_range(start, count, f),
            EncodedData::Huffman(data) => data.for_each_sample_range(start, count, f),
        }
    }

    pub fn to_raw_deviation_data(&self) -> DeviationData {
        match self {
            EncodedData::Normal(data) => data.clone(),
            EncodedData::Huffman(data) => data.to_deviation_data().expect(
                "Huffman encoded data should be valid when converting to raw deviation data",
            ),
            EncodedData::RleOffset(data) => data.to_deviation_data().expect(
                "RLE offset encoded data should be valid when converting to raw deviation data",
            ),
        }
    }
}

pub(super) fn build_deviation_ranges(
    base_bit_mask: &BitSlice<usize, Lsb0>,
    chunk_size: usize,
    num_deviation_bits: usize,
) -> Vec<(usize, usize)> {
    let mut deviation_positions = Vec::with_capacity(num_deviation_bits);
    for bit_pos in 0..chunk_size {
        if !base_bit_mask[bit_pos] {
            deviation_positions.push(bit_pos);
        }
    }

    let mut deviation_ranges: Vec<(usize, usize)> = Vec::new();
    if let Some(&first_pos) = deviation_positions.first() {
        let mut range_start = first_pos;
        let mut prev = first_pos;
        for &pos in deviation_positions.iter().skip(1) {
            if pos == prev + 1 {
                prev = pos;
            } else {
                deviation_ranges.push((range_start, prev + 1));
                range_start = pos;
                prev = pos;
            }
        }
        deviation_ranges.push((range_start, prev + 1));
    }

    deviation_ranges
}

pub(super) fn derive_symbol_layout(
    input: &PreEncodeContext,
) -> (usize, usize, Vec<(usize, usize)>) {
    let chunk_size = input.bit_data.chunk_size();
    let selected_positions = input.layout.selected_base_bit_positions();
    let num_deviation_bits = chunk_size.saturating_sub(selected_positions.len());
    let l_id = bits_needed_nonzero(input.variable_base_table.len());

    let mut base_bit_mask = BitVec::repeat(false, chunk_size);
    for &bit_pos in &selected_positions {
        if bit_pos < chunk_size {
            base_bit_mask.set(bit_pos, true);
        }
    }
    let deviation_ranges =
        build_deviation_ranges(base_bit_mask.as_bitslice(), chunk_size, num_deviation_bits);

    (num_deviation_bits, l_id, deviation_ranges)
}

pub(super) fn build_id_bits_per_base(l_id: usize, num_bases: usize) -> Vec<BitVec<usize, Lsb0>> {
    let mut id_bits_per_base = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = BitVec::repeat(false, l_id);
            id_bits.as_mut_bitslice().store_le(id);
            id_bits_per_base.push(id_bits);
        }
    }

    id_bits_per_base
}

pub(super) fn encode_rows_as_symbol_stream(
    input: &PreEncodeContext,
    num_deviation_bits: usize,
    l_id: usize,
    deviation_ranges: &[(usize, usize)],
    id_bits_per_base: &[BitVec<usize, Lsb0>],
) -> BitVec<usize, Lsb0> {
    let symbol_width = num_deviation_bits + l_id;
    let num_rows = input.bit_data.num_rows();
    let chunk_size = (num_rows / (rayon::current_num_threads() * 4)).clamp(256, 4096);

    let chunk_results: Vec<BitVec<usize, Lsb0>> = (0..num_rows)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|row_chunk| {
            let mut chunk_stream = BitVec::with_capacity(row_chunk.len() * symbol_width);
            for row in row_chunk {
                let id = input.row_to_base_id[row];
                let chunk = unsafe { input.bit_data.get_chunk_unchecked(row) };
                for &(start, end) in deviation_ranges {
                    chunk_stream.extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
                }
                if l_id > 0 {
                    chunk_stream.extend_from_bitslice(id_bits_per_base[id].as_bitslice());
                }
            }
            chunk_stream
        })
        .collect();

    let mut symbol_stream = BitVec::with_capacity(num_rows * symbol_width);
    for chunk_stream in chunk_results {
        symbol_stream.extend_from_bitslice(chunk_stream.as_bitslice());
    }
    symbol_stream
}

pub struct EncodeData {}

impl Filter for EncodeData {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format");
        let encoded = EncodedData::Normal(encode_data(&input));
        Ok(build_compressed_data(input, encoded))
    }
}

pub struct EncodeDataOffsetRLE {}

impl Filter for EncodeDataOffsetRLE {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(
            "Encoding data into compressed format (optimized + RLE + row offset)",
        );
        let encoded = EncodedData::RleOffset(encode_data_offset_rle(&input)?);
        Ok(build_compressed_data(input, encoded))
    }
}

pub(super) fn build_compressed_data(
    input: PreEncodeContext,
    encoded_data: EncodedData,
) -> CompressedData {
    let mut compressed = CompressedData::new(encoded_data, input.bit_data.info.clone());
    compressed.base_table = BaseTable::Raw(input.variable_base_table);
    compressed.layout = input.layout;
    compressed.entropy_sorted_column_order = input.entropy_sorted_column_order;
    compressed.condensed_sample_weights = input
        .bit_data
        .info
        .m_condensed_sample_weights()
        .map(|weights| weights.to_vec());
    compressed
}

fn encode_data(input: &PreEncodeContext) -> DeviationData {
    let (num_deviation_bits, l_id, deviation_ranges) = derive_symbol_layout(input);
    let id_bits_per_base = build_id_bits_per_base(l_id, input.variable_base_table.len());
    let encoded_bit_stream = encode_rows_as_symbol_stream(
        input,
        num_deviation_bits,
        l_id,
        &deviation_ranges,
        &id_bits_per_base,
    );

    DeviationData::new(
        encoded_bit_stream,
        input.bit_data.num_rows(),
        num_deviation_bits,
        l_id,
    )
}

fn encode_data_offset_rle(
    input: &PreEncodeContext,
) -> Result<RleDeviationOffsetData, EntroGdError> {
    let (num_deviation_bits, l_id, deviation_ranges) = derive_symbol_layout(input);
    let symbol_width = num_deviation_bits + l_id;
    let num_rows = input.bit_data.num_rows();
    let id_bits_per_base = build_id_bits_per_base(l_id, input.variable_base_table.len());
    let raw_symbol_stream = encode_rows_as_symbol_stream(
        input,
        num_deviation_bits,
        l_id,
        &deviation_ranges,
        &id_bits_per_base,
    );

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&input.bit_data.info)?;

    let mut symbol_stream = BitVec::new();
    let mut rm_values: Vec<(u8, u8)> = Vec::new();
    let mut row_offsets: Vec<(u32, u32)> = Vec::with_capacity(row_count);

    if num_rows == 0 || symbol_width == 0 {
        return RleDeviationOffsetData::new(
            symbol_stream,
            rm_values,
            row_offsets,
            original_num_samples,
            row_width,
            num_rows,
            num_deviation_bits,
            l_id,
        );
    }

    for row_idx in 0..row_count {
        row_offsets.push((
            u32::try_from(rm_values.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "RLE row offset rm index does not fit into u32".to_string(),
            })?,
            u32::try_from(symbol_stream.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "RLE row offset symbol bit index does not fit into u32".to_string(),
            })?,
        ));

        let row_start = row_idx * row_width;
        let row_end = (row_start + row_width).min(original_num_samples);
        let mut i = row_start;

        while i < row_end {
            let current_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

            let mut run_len = 1usize;
            while i + run_len < row_end && run_len < RLE_MAX_RUN_LEN {
                let next_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i + run_len);
                if next_symbol == current_symbol {
                    run_len += 1;
                } else {
                    break;
                }
            }

            let r_encoded: u8;
            if run_len >= 2 {
                r_encoded = (run_len - 1) as u8;
                symbol_stream.extend_from_bitslice(current_symbol);
                i += run_len;
            } else {
                r_encoded = 0;
            }

            let literal_start = i;
            let mut literal_count = 0usize;
            while i < row_end && literal_count < RLE_MAX_CONTROL_VALUE {
                let this_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

                let mut lookahead_run = 1usize;
                while i + lookahead_run < row_end && lookahead_run < RLE_MAX_RUN_LEN {
                    let lookahead_symbol =
                        symbol_slice(&raw_symbol_stream, symbol_width, i + lookahead_run);
                    if lookahead_symbol == this_symbol {
                        lookahead_run += 1;
                    } else {
                        break;
                    }
                }

                if lookahead_run >= 2 {
                    break;
                }

                symbol_stream.extend_from_bitslice(this_symbol);
                literal_count += 1;
                i += 1;
            }

            if r_encoded == 0 && literal_count == 0 {
                symbol_stream.extend_from_bitslice(current_symbol);
                literal_count = 1;
                i = literal_start + 1;
            }

            rm_values.push((r_encoded, literal_count as u8));
        }
    }

    if original_num_samples < num_rows {
        let mut i = original_num_samples;
        while i < num_rows {
            let current_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

            let mut run_len = 1usize;
            while i + run_len < num_rows && run_len < RLE_MAX_RUN_LEN {
                let next_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i + run_len);
                if next_symbol == current_symbol {
                    run_len += 1;
                } else {
                    break;
                }
            }

            let r_encoded: u8;
            if run_len >= 2 {
                r_encoded = (run_len - 1) as u8;
                symbol_stream.extend_from_bitslice(current_symbol);
                i += run_len;
            } else {
                r_encoded = 0;
            }

            let literal_start = i;
            let mut literal_count = 0usize;
            while i < num_rows && literal_count < RLE_MAX_CONTROL_VALUE {
                let this_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

                let mut lookahead_run = 1usize;
                while i + lookahead_run < num_rows && lookahead_run < RLE_MAX_RUN_LEN {
                    let lookahead_symbol =
                        symbol_slice(&raw_symbol_stream, symbol_width, i + lookahead_run);
                    if lookahead_symbol == this_symbol {
                        lookahead_run += 1;
                    } else {
                        break;
                    }
                }

                if lookahead_run >= 2 {
                    break;
                }

                symbol_stream.extend_from_bitslice(this_symbol);
                literal_count += 1;
                i += 1;
            }

            if r_encoded == 0 && literal_count == 0 {
                symbol_stream.extend_from_bitslice(current_symbol);
                literal_count = 1;
                i = literal_start + 1;
            }

            rm_values.push((r_encoded, literal_count as u8));
        }
    }

    RleDeviationOffsetData::new(
        symbol_stream,
        rm_values,
        row_offsets,
        original_num_samples,
        row_width,
        num_rows,
        num_deviation_bits,
        l_id,
    )
}

fn symbol_slice(
    symbol_stream: &BitVec<usize, Lsb0>,
    symbol_width: usize,
    row: usize,
) -> &BitSlice<usize, Lsb0> {
    let start = row * symbol_width;
    let end = start + symbol_width;
    debug_assert!(end <= symbol_stream.len());
    unsafe { symbol_stream.get_unchecked(start..end) }
}
