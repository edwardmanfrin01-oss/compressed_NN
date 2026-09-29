use bitvec::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::encoding_core::{
    DeviationData, DeviationSample, DeviationSampleRef, EncodedData, HuffmanDeviationData,
    build_compressed_data, derive_symbol_layout, huffman_row_layout,
};

use crate::compression::base_table::PreEncodeContext;
use crate::compression::data::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

const HUFFMAN_CODE_LENGTH_BITS: usize = 5;
const HUFFMAN_MAX_CODE_LENGTH: u8 = 24;

pub struct EncodeDataHuffman {}

impl Filter for EncodeDataHuffman {
    type Input = PreEncodeContext;
    type Output = super::encoding_core::CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Encoding data into compressed format (Huffman base-id only)");
        let encoded = EncodedData::Huffman(encode_data_huffman(&input.bit_data, &input)?);
        Ok(build_compressed_data(input, encoded))
    }
}

pub(super) fn encode_data_huffman(
    bit_data: &BitDataSet,
    input: &PreEncodeContext,
) -> Result<HuffmanDeviationData, EntroGdError> {
    let (num_deviation_bits, l_id, deviation_ranges) = derive_symbol_layout(input);
    let frequencies: Vec<SymbolFrequency> = input
        .variable_base_table
        .iter()
        .enumerate()
        .map(|(id, (_base, count))| SymbolFrequency {
            symbol: id as u64,
            frequency: *count,
        })
        .collect();
    let (canonical_symbols, canonical_code_lengths) = build_canonical_huffman_table(frequencies)?;
    let codes_by_symbol = build_huffman_code_map(&canonical_symbols, &canonical_code_lengths)?;

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&bit_data.info)?;

    let num_rows = bit_data.num_rows();
    let effective_row_width = if row_width > 0 { row_width } else { 1 };
    let target = (num_rows / (rayon::current_num_threads() * 4)).clamp(256, 4096);
    let chunk_size = target.div_ceil(effective_row_width) * effective_row_width;

    struct ChunkResult {
        pixel_bits: BitVec<usize, Lsb0>,
        deviation_bits: BitVec<usize, Lsb0>,
        row_offsets: Vec<usize>,
    }

    let _timer = ScopedTimer::info("Encoding pixel bit stream");
    let par_chunk_results: Result<Vec<ChunkResult>, EntroGdError> = (0..num_rows)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk_indices| -> Result<ChunkResult, EntroGdError> {
            let mut pixel_bits = BitVec::new();
            let mut deviation_bits =
                BitVec::with_capacity(chunk_indices.len() * num_deviation_bits);
            let mut offsets = Vec::new();

            for sample_idx in chunk_indices {
                if sample_idx < original_num_samples && sample_idx % effective_row_width == 0 {
                    offsets.push(pixel_bits.len());
                }
                let chunk = unsafe { bit_data.get_chunk_unchecked(sample_idx) };
                for &(start, end) in &deviation_ranges {
                    deviation_bits.extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
                }
                let symbol = input.row_to_base_id[sample_idx] as u64;
                let idx = symbol as usize;
                let entry =
                    codes_by_symbol
                        .get(idx)
                        .ok_or_else(|| EntroGdError::InvalidMetadata {
                            message: format!(
                                "Huffman symbol {} exceeds symbol table bounds",
                                symbol
                            ),
                        })?;
                let (code, code_len) = entry.ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for base-id symbol {}", symbol),
                })?;
                append_code_bits(&mut pixel_bits, code, code_len);
            }
            Ok(ChunkResult {
                pixel_bits,
                deviation_bits,
                row_offsets: offsets,
            })
        })
        .collect();
    drop(_timer);
    let par_chunk_results = par_chunk_results?;

    let total_pixel_bits: usize = par_chunk_results.iter().map(|c| c.pixel_bits.len()).sum();
    let mut pixel_bit_stream = BitVec::with_capacity(total_pixel_bits);
    let mut raw_deviation_bit_stream = BitVec::with_capacity(num_rows * num_deviation_bits);
    let mut row_offsets: Vec<u32> = Vec::with_capacity(row_count);

    let mut prefix_pixel_len = 0usize;
    for chunk in par_chunk_results {
        for local_offset in chunk.row_offsets {
            let global_offset = prefix_pixel_len + local_offset;
            row_offsets.push(u32::try_from(global_offset).map_err(|_| {
                EntroGdError::InvalidMetadata {
                    message: "Huffman row offset does not fit into u32
                    Size: of offset much mean that input is above 536 MB
                    Change how offsets are stored to allow larger inputs"
                        .to_string(),
                }
            })?);
        }
        prefix_pixel_len += chunk.pixel_bits.len();
        pixel_bit_stream.extend_from_bitslice(chunk.pixel_bits.as_bitslice());
        raw_deviation_bit_stream.extend_from_bitslice(chunk.deviation_bits.as_bitslice());
    }
    tracing::debug!(row_offeset_size = ?row_offsets.len(), "Encoded Huffman row offsets");

    HuffmanDeviationData::new(
        pixel_bit_stream,
        raw_deviation_bit_stream,
        canonical_symbols,
        canonical_code_lengths,
        row_offsets,
        bit_data.num_rows(),
        original_num_samples,
        num_deviation_bits,
        l_id,
        row_width,
    )
}

#[derive(Debug, Clone, Copy)]
struct HuffmanCode {
    symbol: u64,
    code: u32,
    len: u8,
}

impl HuffmanDeviationData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pixel_bit_stream: BitVec<usize, Lsb0>,
        raw_deviation_bit_stream: BitVec<usize, Lsb0>,
        canonical_symbols: Vec<u64>,
        canonical_code_lengths: Vec<u8>,
        row_offsets: Vec<u32>,
        num_samples: usize,
        original_num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
        row_width: usize,
    ) -> Result<Self, EntroGdError> {
        if canonical_symbols.len() != canonical_code_lengths.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman symbol and length tables differ in size".to_string(),
            });
        }
        if original_num_samples > num_samples {
            return Err(EntroGdError::InvalidMetadata {
                message: "original sample count exceeds total sample count".to_string(),
            });
        }
        if original_num_samples > 0 && row_width == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "row width must be > 0 when original samples are present".to_string(),
            });
        }

        let expected_row_count = if original_num_samples == 0 {
            0
        } else {
            original_num_samples.div_ceil(row_width)
        };

        if row_offsets.len() != expected_row_count {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "row offset count mismatch: expected {}, got {}",
                    expected_row_count,
                    row_offsets.len()
                ),
            });
        }

        let symbol_width = num_id_bits;
        if symbol_width > 64 {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "Huffman symbol width {} exceeds supported 64-bit range",
                    symbol_width
                ),
            });
        }

        let expected_raw_deviation_len =
            num_samples.checked_mul(num_deviation_bits).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw deviation stream length overflow".to_string(),
                }
            })?;

        if raw_deviation_bit_stream.len() != expected_raw_deviation_len {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "raw deviation stream length mismatch: expected {}, got {}",
                    expected_raw_deviation_len,
                    raw_deviation_bit_stream.len()
                ),
            });
        }

        let max_symbol = if symbol_width >= 64 {
            u64::MAX
        } else {
            (1u64 << symbol_width) - 1
        };
        for &symbol in &canonical_symbols {
            if symbol > max_symbol {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "Huffman symbol {} exceeds declared symbol width {}",
                        symbol, symbol_width
                    ),
                });
            }
        }

        let codes = rebuild_huffman_codes(&canonical_symbols, &canonical_code_lengths)?;
        let max_code_length = codes.iter().map(|code| code.len).max().unwrap_or(0);

        let mut min_code_by_len = vec![u32::MAX; max_code_length as usize + 1];
        let mut max_code_by_len = vec![0u32; max_code_length as usize + 1];
        let mut first_symbol_index_by_len = vec![usize::MAX; max_code_length as usize + 1];
        let mut canonical_symbols_by_len: Vec<Vec<u64>> =
            vec![Vec::new(); max_code_length as usize + 1];

        for code in &codes {
            canonical_symbols_by_len[code.len as usize].push(code.symbol);
        }

        let mut current_index = 0usize;
        let mut symbols_in_order: Vec<u64> = Vec::with_capacity(canonical_symbols.len());
        for len in 1..=max_code_length as usize {
            symbols_in_order.extend(canonical_symbols_by_len[len].clone());
            if !canonical_symbols_by_len[len].is_empty() {
                let codes_at_len: Vec<u32> = codes
                    .iter()
                    .filter(|c| c.len as usize == len)
                    .map(|c| c.code)
                    .collect();

                if let Some(&first_code) = codes_at_len.first() {
                    min_code_by_len[len] = first_code;
                }
                if let Some(&last_code) = codes_at_len.last() {
                    max_code_by_len[len] = last_code;
                }
                first_symbol_index_by_len[len] = current_index;
                current_index += canonical_symbols_by_len[len].len();
            }
        }

        if num_samples > 0 && codes.is_empty() {
            return Err(EntroGdError::InvalidMetadata {
                message: "non-empty Huffman stream is missing a symbol table".to_string(),
            });
        }

        let mut previous_offset = 0usize;
        for &offset in &row_offsets {
            let current_offset = offset as usize;
            if current_offset < previous_offset {
                return Err(EntroGdError::InvalidMetadata {
                    message: "row offsets must be non-decreasing".to_string(),
                });
            }
            if current_offset > pixel_bit_stream.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "row offset {} exceeds Huffman pixel stream length {}",
                        current_offset,
                        pixel_bit_stream.len()
                    ),
                });
            }
            previous_offset = current_offset;
        }

        let mut data = HuffmanDeviationData {
            pixel_bit_stream,
            raw_deviation_bit_stream,
            canonical_symbols,
            canonical_code_lengths,
            row_offsets,
            num_samples,
            original_num_samples,
            num_deviation_bits,
            num_id_bits,
            row_width,
            max_code_length,
            min_code_by_len,
            max_code_by_len,
            first_symbol_index_by_len,
            decode_symbols: symbols_in_order,
            original_stream_end_offset_bits: 0,
        };

        data.original_stream_end_offset_bits = if data.original_num_samples == 0 {
            0
        } else {
            let last_row_offset = data.row_offsets[data.row_offsets.len() - 1] as usize;
            let last_row_len = data.original_num_samples
                .saturating_sub((data.row_offsets.len().saturating_sub(1)) * data.row_width);
            data.advance_by_symbols(last_row_offset, last_row_len)?
        };

        if data.original_stream_end_offset_bits > data.pixel_bit_stream.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "decoded original stream length exceeds Huffman bitstream size"
                    .to_string(),
            });
        }

        Ok(data)
    }

    pub fn pixel_bit_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.pixel_bit_stream
    }

    pub fn raw_deviation_bit_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.raw_deviation_bit_stream
    }

    pub fn canonical_symbols(&self) -> &[u64] {
        &self.canonical_symbols
    }

    pub fn canonical_code_lengths(&self) -> &[u8] {
        &self.canonical_code_lengths
    }

    pub fn row_offsets(&self) -> &[u32] {
        &self.row_offsets
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn original_num_samples(&self) -> usize {
        self.original_num_samples
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }

        let (start_bit, symbols_to_decode) = if sample_idx < self.original_num_samples {
            let row_idx = sample_idx / self.row_width;
            let col_idx = sample_idx % self.row_width;
            (
                *self.row_offsets.get(row_idx)? as usize,
                col_idx.saturating_add(1),
            )
        } else {
            (
                self.original_stream_end_offset_bits,
                sample_idx
                    .checked_sub(self.original_num_samples)?
                    .saturating_add(1),
            )
        };

        let mut bit_pos = start_bit;
        let mut symbol = 0u64;
        for _ in 0..symbols_to_decode {
            let (decoded_symbol, decoded_len) = self.decode_one(bit_pos).ok()?;
            symbol = decoded_symbol;
            bit_pos += decoded_len;
        }

        let start = sample_idx.checked_mul(self.num_deviation_bits)?;
        let end = start.checked_add(self.num_deviation_bits)?;
        debug_assert!(end <= self.raw_deviation_bit_stream.len());
        let deviation = unsafe {
            self.raw_deviation_bit_stream
                .get_unchecked(start..end)
                .to_bitvec()
        };
        let id_value = symbol;

        Some(DeviationSample {
            deviation,
            id: bitvec_from_u64(id_value, self.num_id_bits),
        })
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

    pub fn get_encoded_size(&self) -> usize {
        let symbol_width = self.num_id_bits;
        16 + 8
            + 8
            + 32
            + self.row_offsets.len() * 32
            + self.canonical_symbols.len() * (symbol_width + HUFFMAN_CODE_LENGTH_BITS)
            + self.raw_deviation_bit_stream.len()
            + self.pixel_bit_stream.len()
    }

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let capped_limit = limit.min(self.num_samples);
        if capped_limit == 0 {
            return Ok(());
        }

        let mut bit_pos = 0usize;
        let mut id_bits_buffer = BitVec::with_capacity(self.num_id_bits);

        for sample_idx in 0..capped_limit {
            let (symbol, consumed_bits) = self.decode_one(bit_pos)?;
            bit_pos += consumed_bits;

            let deviation_start = sample_idx * self.num_deviation_bits;
            let deviation_end = deviation_start + self.num_deviation_bits;
            let deviation_bits = unsafe {
                self.raw_deviation_bit_stream
                    .get_unchecked(deviation_start..deviation_end)
            };

            id_bits_buffer.clear();
            append_symbol_bits(&mut id_bits_buffer, symbol, self.num_id_bits);

            f(DeviationSampleRef {
                deviation: deviation_bits,
                id: id_bits_buffer.as_bitslice(),
            })?;
        }

        Ok(())
    }

    pub(crate) fn for_each_sample_at_sorted_indices(
        &self,
        sorted_indices: &[usize],
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        if sorted_indices.is_empty() {
            return Ok(());
        }

        let mut id_bits_buffer = BitVec::with_capacity(self.num_id_bits);
        let mut req_ptr = 0usize;

        // Process original samples row by row, seeking to the row's bit offset.
        if self.row_width > 0 {
            while req_ptr < sorted_indices.len()
                && sorted_indices[req_ptr] < self.original_num_samples
            {
                let sample_idx = sorted_indices[req_ptr];
                let row_idx = sample_idx / self.row_width;
                let row_start_sample = row_idx * self.row_width;
                let row_end = row_start_sample + self.row_width;

                let mut bit_pos =
                    *self
                        .row_offsets
                        .get(row_idx)
                        .ok_or(EntroGdError::InvalidMetadata {
                            message: format!("Huffman row offset missing for row {}", row_idx),
                        })? as usize;
                let mut col = 0usize;
                let mut last_symbol = 0u64;

                // Advance symbol by symbol through this row, yielding at each requested column.
                while req_ptr < sorted_indices.len() && sorted_indices[req_ptr] < row_end {
                    let target_col = sorted_indices[req_ptr] - row_start_sample;

                    // Decode symbols up to and including target_col.
                    while col <= target_col {
                        let (sym, consumed) = self.decode_one(bit_pos)?;
                        last_symbol = sym;
                        bit_pos += consumed;
                        col += 1;
                    }

                    let abs_sample_idx = row_start_sample + target_col;
                    let deviation_start = abs_sample_idx * self.num_deviation_bits;
                    let deviation_end = deviation_start + self.num_deviation_bits;
                    let deviation_bits = unsafe {
                        self.raw_deviation_bit_stream
                            .get_unchecked(deviation_start..deviation_end)
                    };

                    id_bits_buffer.clear();
                    append_symbol_bits(&mut id_bits_buffer, last_symbol, self.num_id_bits);

                    // Yield once (and again for any duplicate indices at the same position).
                    while req_ptr < sorted_indices.len()
                        && sorted_indices[req_ptr] < row_end
                        && sorted_indices[req_ptr] - row_start_sample == target_col
                    {
                        f(DeviationSampleRef {
                            deviation: deviation_bits,
                            id: id_bits_buffer.as_bitslice(),
                        })?;
                        req_ptr += 1;
                    }
                }
            }
        }

        // Fall back to get_sample for any padded samples (>= original_num_samples).
        while req_ptr < sorted_indices.len() {
            let sample_idx = sorted_indices[req_ptr];
            let sample = self
                .get_sample(sample_idx)
                .ok_or(EntroGdError::DecompressionSampleMissing { sample_idx })?;
            f(DeviationSampleRef {
                deviation: sample.deviation.as_bitslice(),
                id: sample.id.as_bitslice(),
            })?;
            req_ptr += 1;
        }

        Ok(())
    }

    pub(crate) fn for_each_sample_range(
        &self,
        start: usize,
        count: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let capped_count = count.min(self.num_samples.saturating_sub(start));
        if capped_count == 0 {
            return Ok(());
        }

        // Seek to the row boundary closest to `start`, then skip within-row symbols.
        let (mut bit_pos, col_skip) = if self.row_width > 0 && start < self.original_num_samples {
            let row_idx = start / self.row_width;
            let col_offset = start % self.row_width;
            let row_bit_pos =
                *self
                    .row_offsets
                    .get(row_idx)
                    .ok_or(EntroGdError::InvalidMetadata {
                        message: format!("Huffman row offset missing for row {}", row_idx),
                    })? as usize;
            (row_bit_pos, col_offset)
        } else {
            // Padded samples: start from end of original stream, skip forward.
            (
                self.original_stream_end_offset_bits,
                start.saturating_sub(self.original_num_samples),
            )
        };

        // Skip col_skip symbols to reach the exact starting sample.
        for _ in 0..col_skip {
            let (_, consumed) = self.decode_one(bit_pos)?;
            bit_pos += consumed;
        }

        let mut id_bits_buffer = BitVec::with_capacity(self.num_id_bits);

        for sample_idx in start..start + capped_count {
            let (symbol, consumed) = self.decode_one(bit_pos)?;
            bit_pos += consumed;

            let deviation_start = sample_idx * self.num_deviation_bits;
            let deviation_end = deviation_start + self.num_deviation_bits;
            let deviation_bits = unsafe {
                self.raw_deviation_bit_stream
                    .get_unchecked(deviation_start..deviation_end)
            };

            id_bits_buffer.clear();
            append_symbol_bits(&mut id_bits_buffer, symbol, self.num_id_bits);

            f(DeviationSampleRef {
                deviation: deviation_bits,
                id: id_bits_buffer.as_bitslice(),
            })?;
        }

        Ok(())
    }

    pub fn to_deviation_data(&self) -> Result<DeviationData, EntroGdError> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let expected_bits = self.num_samples.checked_mul(symbol_width).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "decoded Huffman deviation stream length overflow".to_string(),
            }
        })?;
        let mut raw = BitVec::with_capacity(expected_bits);
        let mut bit_pos = 0usize;

        for sample_idx in 0..self.num_samples {
            let (symbol, consumed_bits) = self.decode_one(bit_pos)?;

            let deviation_start = sample_idx * self.num_deviation_bits;
            let deviation_end = deviation_start + self.num_deviation_bits;
            raw.extend_from_bitslice(
                &self.raw_deviation_bit_stream[deviation_start..deviation_end],
            );

            append_symbol_bits(&mut raw, symbol, self.num_id_bits);
            bit_pos += consumed_bits;
        }

        Ok(DeviationData::new(
            raw,
            self.num_samples,
            self.num_deviation_bits,
            self.num_id_bits,
        ))
    }

    fn advance_by_symbols(
        &self,
        start_bit: usize,
        symbol_count: usize,
    ) -> Result<usize, EntroGdError> {
        let mut bit_pos = start_bit;
        for _ in 0..symbol_count {
            let (_, consumed_bits) = self.decode_one(bit_pos)?;
            bit_pos += consumed_bits;
        }
        Ok(bit_pos)
    }

    fn decode_one(&self, bit_pos: usize) -> Result<(u64, usize), EntroGdError> {
        debug_assert!(
            self.max_code_length != 0,
            "cannot decode from an empty Huffman code table"
        );
        let mut code = 0u32;
        for len in 1..=self.max_code_length as usize {
            let next_bit_pos = bit_pos + len - 1;
            if next_bit_pos >= self.pixel_bit_stream.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: "unexpected end of Huffman pixel stream".to_string(),
                });
            }

            code = (code << 1)
                | u32::from(unsafe { *self.pixel_bit_stream.get_unchecked(next_bit_pos) });

            let min_code = self.min_code_by_len[len];
            let max_code = self.max_code_by_len[len];
            if min_code != u32::MAX && code >= min_code && code <= max_code {
                let symbol_index = self.first_symbol_index_by_len[len] + (code - min_code) as usize;
                if let Some(&symbol) = self.decode_symbols.get(symbol_index) {
                    return Ok((symbol, len));
                }
            }
        }

        Err(EntroGdError::InvalidMetadata {
            message: "failed to decode Huffman symbol from bitstream".to_string(),
        })
    }
}

fn rebuild_huffman_codes(
    canonical_symbols: &[u64],
    canonical_code_lengths: &[u8],
) -> Result<Vec<HuffmanCode>, EntroGdError> {
    if canonical_symbols.is_empty() {
        return Ok(Vec::new());
    }

    let mut codes = Vec::with_capacity(canonical_symbols.len());
    let mut next_code = 0u32;
    let mut previous_len = 0u8;
    let mut previous_symbol = None;

    for (&symbol, &len) in canonical_symbols.iter().zip(canonical_code_lengths.iter()) {
        if !(1..=HUFFMAN_MAX_CODE_LENGTH).contains(&len) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("invalid Huffman code length {}", len),
            });
        }

        if let Some((prev_len, prev_symbol_value)) = previous_symbol
            && (len < prev_len || (len == prev_len && symbol <= prev_symbol_value))
        {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman canonical table is not sorted by (length, symbol)".to_string(),
            });
        }

        next_code <<= (len - previous_len) as u32;
        if (next_code as u64) >= (1u64 << len) {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman canonical table violates prefix-code bounds".to_string(),
            });
        }

        codes.push(HuffmanCode {
            symbol,
            code: next_code,
            len,
        });

        previous_len = len;
        previous_symbol = Some((len, symbol));
        next_code = next_code.saturating_add(1);
    }

    Ok(codes)
}

fn build_huffman_code_map(
    canonical_symbols: &[u64],
    canonical_code_lengths: &[u8],
) -> Result<Vec<Option<(u32, u8)>>, EntroGdError> {
    let codes = rebuild_huffman_codes(canonical_symbols, canonical_code_lengths)?;
    let max_symbol = canonical_symbols.iter().copied().max().unwrap_or(0) as usize;
    let mut codes_by_symbol = vec![None; max_symbol + 1];

    for code in codes {
        let idx = code.symbol as usize;
        if codes_by_symbol[idx].is_some() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("duplicate Huffman symbol {}", code.symbol),
            });
        }
        codes_by_symbol[idx] = Some((code.code, code.len));
    }
    Ok(codes_by_symbol)
}

fn bitvec_from_u64(value: u64, width: usize) -> BitVec<usize, Lsb0> {
    let mut bits = BitVec::with_capacity(width);
    append_symbol_bits(&mut bits, value, width);
    bits
}

fn append_symbol_bits(out: &mut BitVec<usize, Lsb0>, value: u64, width: usize) {
    for shift in 0..width {
        out.push(((value >> shift) & 1) == 1);
    }
}

fn build_canonical_huffman_table(
    mut symbol_frequencies: Vec<SymbolFrequency>,
) -> Result<(Vec<u64>, Vec<u8>), EntroGdError> {
    if symbol_frequencies.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if symbol_frequencies.len() > u32::MAX as usize {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "Huffman symbol table has {} entries, exceeding u32::MAX",
                symbol_frequencies.len()
            ),
        });
    }
    symbol_frequencies.sort_unstable_by(|a, b| a.symbol.cmp(&b.symbol));

    if symbol_frequencies.len() == 1 {
        return Ok((vec![symbol_frequencies[0].symbol], vec![1]));
    }

    let raw_lengths = build_huffman_code_lengths(&symbol_frequencies)?;

    let mut canonical_entries: Vec<(u64, u8)> = symbol_frequencies
        .iter()
        .zip(raw_lengths.iter())
        .map(|(symbol, &len)| (symbol.symbol, len as u8))
        .collect();
    canonical_entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    let (canonical_symbols, canonical_lengths): (Vec<u64>, Vec<u8>) =
        canonical_entries.into_iter().unzip();
    Ok((canonical_symbols, canonical_lengths))
}

fn build_huffman_code_lengths(symbols: &[SymbolFrequency]) -> Result<Vec<usize>, EntroGdError> {
    let mut nodes: Vec<HuffmanNode> = symbols
        .iter()
        .map(|symbol| HuffmanNode {
            weight: symbol.frequency,
            min_symbol: symbol.symbol,
            parent: None,
        })
        .collect();

    let mut heap = BinaryHeap::<Reverse<(usize, u64, usize)>>::new();
    for (idx, node) in nodes.iter().enumerate() {
        heap.push(Reverse((node.weight, node.min_symbol, idx)));
    }

    while heap.len() > 1 {
        let Reverse((left_weight, left_min_symbol, left_idx)) = heap.pop().unwrap();
        let Reverse((right_weight, right_min_symbol, right_idx)) = heap.pop().unwrap();
        let parent_idx = nodes.len();
        nodes[left_idx].parent = Some(parent_idx);
        nodes[right_idx].parent = Some(parent_idx);
        nodes.push(HuffmanNode {
            weight: left_weight.checked_add(right_weight).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "Huffman frequency sum overflow".to_string(),
                }
            })?,
            min_symbol: left_min_symbol.min(right_min_symbol),
            parent: None,
        });
        heap.push(Reverse((
            left_weight
                .checked_add(right_weight)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "Huffman frequency sum overflow".to_string(),
                })?,
            left_min_symbol.min(right_min_symbol),
            parent_idx,
        )));
    }

    let mut lengths = Vec::with_capacity(symbols.len());
    for leaf_idx in 0..symbols.len() {
        let mut depth = 0usize;
        let mut current = leaf_idx;
        while let Some(parent) = nodes[current].parent {
            depth += 1;
            current = parent;
        }
        lengths.push(depth.max(1));
    }

    Ok(lengths)
}

fn append_code_bits(out: &mut BitVec<usize, Lsb0>, code: u32, code_len: u8) {
    for shift in (0..code_len as usize).rev() {
        out.push(((code >> shift) & 1) == 1);
    }
}

#[derive(Debug, Clone, Copy)]
struct SymbolFrequency {
    symbol: u64,
    frequency: usize,
}

#[derive(Debug, Clone, Copy)]
struct HuffmanNode {
    weight: usize,
    min_symbol: u64,
    parent: Option<usize>,
}
