use super::encoding_core::{
    DeviationData, DeviationSample, DeviationSampleRef, RleDeviationOffsetData,
};
use crate::{ScopedTimer, error::EntroGdError};
use bitvec::prelude::*;

pub(crate) const RLE_SHORT_MAX: u8 = 7;
pub(crate) const RLE_LONG_MIN: u8 = 8;
pub(crate) const RLE_LONG_MAX: u8 = 135;

pub(super) fn write_rle_control_value(out: &mut BitVec<usize, Lsb0>, value: u8) {
    assert!(
        value <= RLE_LONG_MAX,
        "RLE control value {} out of range (max {})",
        value,
        RLE_LONG_MAX
    );

    if value <= RLE_SHORT_MAX {
        // 4-bit packet: 0 + 3-bit payload
        out.push(false);
        for shift in 0..3 {
            out.push(((value >> shift) & 1) == 1);
        }
    } else {
        // 8-bit packet: 1 + 7-bit payload with bias -8 (stored payload = value - 8)
        out.push(true);
        let payload = value - RLE_LONG_MIN;
        for shift in 0..7 {
            out.push(((payload >> shift) & 1) == 1);
        }
    }
}

impl RleDeviationOffsetData {
    pub fn new(
        symbol_bit_stream: BitVec<usize, Lsb0>,
        rm_values: Vec<(u8, u8)>,
        row_offsets: Vec<(u32, u32)>,
        original_num_samples: usize,
        row_width: usize,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Result<Self, EntroGdError> {
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

        if expected_row_count > 0 && row_offsets.first().copied() != Some((0, 0)) {
            return Err(EntroGdError::InvalidMetadata {
                message: "first row offset must start at rm index 0 and symbol bit 0".to_string(),
            });
        }

        let mut previous = (0u32, 0u32);
        for &(rm_idx, symbol_bit_idx) in &row_offsets {
            if (rm_idx, symbol_bit_idx) < previous {
                return Err(EntroGdError::InvalidMetadata {
                    message: "row offsets must be non-decreasing".to_string(),
                });
            }
            if rm_idx as usize > rm_values.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "row offset rm index {} exceeds rm value count {}",
                        rm_idx,
                        rm_values.len()
                    ),
                });
            }
            if symbol_bit_idx as usize > symbol_bit_stream.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "row offset symbol bit {} exceeds symbol stream length {}",
                        symbol_bit_idx,
                        symbol_bit_stream.len()
                    ),
                });
            }
            previous = (rm_idx, symbol_bit_idx);
        }

        let mut rm_control_stream = BitVec::new();
        for &(r, m) in &rm_values {
            write_rle_control_value(&mut rm_control_stream, r);
            write_rle_control_value(&mut rm_control_stream, m);
        }

        let mut row_offset_stream = BitVec::with_capacity(row_offsets.len() * 64);
        for &(rm_idx, symbol_bit_idx) in &row_offsets {
            append_u32_bits(&mut row_offset_stream, rm_idx);
            append_u32_bits(&mut row_offset_stream, symbol_bit_idx);
        }

        Ok(RleDeviationOffsetData {
            symbol_bit_stream,
            rm_values,
            rm_control_stream,
            row_offset_stream,
            row_offsets,
            original_num_samples,
            row_width,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        })
    }

    pub fn symbol_bit_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.symbol_bit_stream
    }

    pub fn rm_control_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.rm_control_stream
    }

    pub fn row_offset_stream(&self) -> &BitVec<usize, Lsb0> {
        &self.row_offset_stream
    }

    pub fn rm_values(&self) -> &[(u8, u8)] {
        &self.rm_values
    }

    pub fn row_offsets(&self) -> &[(u32, u32)] {
        &self.row_offsets
    }

    pub fn original_num_samples(&self) -> usize {
        self.original_num_samples
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }

        if self.row_width > 0 && sample_idx < self.original_num_samples {
            let row_idx = sample_idx / self.row_width;
            let col_idx = sample_idx % self.row_width;
            let (start_rm_idx, start_symbol_bit_idx) = *self.row_offsets.get(row_idx)?;
            return self.decode_sample_from(
                start_rm_idx as usize,
                start_symbol_bit_idx as usize,
                col_idx,
            );
        }

        self.decode_sample_from(0, 0, sample_idx)
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
        32 + 32
            + self.row_offset_stream.len()
            + self.rm_control_stream.len()
            + self.symbol_bit_stream.len()
    }

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let capped_limit = limit.min(self.num_samples);
        if capped_limit == 0 {
            return Ok(());
        }

        let mut symbol_cursor = 0usize;
        let mut decoded_samples = 0usize;

        for &(r_encoded, m_count) in &self.rm_values {
            if decoded_samples >= capped_limit {
                break;
            }

            if r_encoded > 0 {
                let run_len = (r_encoded as usize) + 1;
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE offset run symbol exceeds symbol stream length".to_string(),
                    });
                }

                let run_symbol = unsafe {
                    self.symbol_bit_stream
                        .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                };
                let sample = DeviationSampleRef {
                    deviation: unsafe { run_symbol.get_unchecked(..self.num_deviation_bits) },
                    id: unsafe { run_symbol.get_unchecked(self.num_deviation_bits..) },
                };

                let remaining = capped_limit - decoded_samples;
                let to_emit = run_len.min(remaining);
                for _ in 0..to_emit {
                    f(DeviationSampleRef {
                        deviation: sample.deviation,
                        id: sample.id,
                    })?;
                }

                decoded_samples += to_emit;
                symbol_cursor += symbol_width;
            }

            if decoded_samples >= capped_limit {
                break;
            }

            let literal_count = m_count as usize;
            let remaining = capped_limit - decoded_samples;
            let to_emit = literal_count.min(remaining);
            for _ in 0..to_emit {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE offset literal symbol exceeds symbol stream length"
                            .to_string(),
                    });
                }

                let literal = unsafe {
                    self.symbol_bit_stream
                        .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                };
                f(DeviationSampleRef {
                    deviation: unsafe { literal.get_unchecked(..self.num_deviation_bits) },
                    id: unsafe { literal.get_unchecked(self.num_deviation_bits..) },
                })?;
                symbol_cursor += symbol_width;
            }
            decoded_samples += to_emit;
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

        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let mut req_ptr = 0usize;

        // Process original samples row by row, jumping to each row's start offset.
        if self.row_width > 0 {
            while req_ptr < sorted_indices.len()
                && sorted_indices[req_ptr] < self.original_num_samples
            {
                let sample_idx = sorted_indices[req_ptr];
                let row_idx = sample_idx / self.row_width;
                let (start_rm_idx, start_sym_bit) = self.row_offsets[row_idx];
                let row_end = (row_idx + 1) * self.row_width;

                let mut symbol_cursor = start_sym_bit as usize;
                let mut logical_idx = 0usize; // col position within this row

                for &(r_encoded, m_count) in self.rm_values.iter().skip(start_rm_idx as usize) {
                    if req_ptr >= sorted_indices.len() || sorted_indices[req_ptr] >= row_end {
                        break;
                    }

                    if r_encoded > 0 {
                        if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                            return Err(EntroGdError::InvalidMetadata {
                                message: "RLE offset run symbol exceeds symbol stream length"
                                    .to_string(),
                            });
                        }
                        let run_len = (r_encoded as usize) + 1;
                        let run_symbol = unsafe {
                            self.symbol_bit_stream
                                .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                        };
                        while req_ptr < sorted_indices.len()
                            && sorted_indices[req_ptr] < row_end
                            && sorted_indices[req_ptr] - (row_idx * self.row_width)
                                < logical_idx + run_len
                        {
                            f(DeviationSampleRef {
                                deviation: unsafe {
                                    run_symbol.get_unchecked(..self.num_deviation_bits)
                                },
                                id: unsafe { run_symbol.get_unchecked(self.num_deviation_bits..) },
                            })?;
                            req_ptr += 1;
                        }
                        symbol_cursor += symbol_width;
                        logical_idx += run_len;
                    }

                    if req_ptr >= sorted_indices.len() || sorted_indices[req_ptr] >= row_end {
                        break;
                    }

                    let literal_count = m_count as usize;
                    while req_ptr < sorted_indices.len()
                        && sorted_indices[req_ptr] < row_end
                        && sorted_indices[req_ptr] - (row_idx * self.row_width)
                            < logical_idx + literal_count
                    {
                        let col = sorted_indices[req_ptr] - (row_idx * self.row_width);
                        let offset = col - logical_idx;
                        let start = symbol_cursor + offset * symbol_width;
                        let end = start + symbol_width;
                        if end > self.symbol_bit_stream.len() {
                            return Err(EntroGdError::InvalidMetadata {
                                message: "RLE offset literal symbol exceeds symbol stream length"
                                    .to_string(),
                            });
                        }
                        let symbol = unsafe { self.symbol_bit_stream.get_unchecked(start..end) };
                        f(DeviationSampleRef {
                            deviation: unsafe { symbol.get_unchecked(..self.num_deviation_bits) },
                            id: unsafe { symbol.get_unchecked(self.num_deviation_bits..) },
                        })?;
                        req_ptr += 1;
                    }
                    symbol_cursor += literal_count * symbol_width;
                    logical_idx += literal_count;

                    if logical_idx >= self.row_width {
                        break;
                    }
                }
            }
        }

        // Fall back to get_sample for any padded samples (>= original_num_samples).
        // Padded samples are rare (partial row padding) so individual lookups are acceptable.
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
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let capped_count = count.min(self.num_samples.saturating_sub(start));
        if capped_count == 0 {
            return Ok(());
        }

        // Jump to the nearest stored row boundary, then skip remaining columns within that row.
        let (start_rm_idx, sym_cursor_init, skip_from_boundary) =
            if self.row_width > 0 && start < self.original_num_samples {
                let row_idx = start / self.row_width;
                let col_offset = start % self.row_width;
                let (rm_start, sym_start) = self.row_offsets[row_idx];
                (rm_start as usize, sym_start as usize, col_offset)
            } else {
                // Padded samples have no stored offset — scan from stream beginning.
                (0, 0, start)
            };

        let mut sym_cursor = sym_cursor_init;
        // `processed` counts samples from the row boundary (or stream start) before emitting.
        let mut processed = 0usize;
        let mut yielded = 0usize;

        for &(r_encoded, m_count) in self.rm_values.iter().skip(start_rm_idx) {
            if yielded >= capped_count {
                break;
            }

            if r_encoded > 0 {
                let run_len = (r_encoded as usize) + 1;
                if sym_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE offset run symbol exceeds symbol stream length".to_string(),
                    });
                }

                // Intersection of this run with the requested window [skip, skip+capped_count).
                let run_start = processed;
                let run_end = processed + run_len;
                let emit_start = run_start.max(skip_from_boundary);
                let emit_end = run_end.min(skip_from_boundary + capped_count);

                if emit_start < emit_end {
                    let run_sym = unsafe {
                        self.symbol_bit_stream
                            .get_unchecked(sym_cursor..sym_cursor + symbol_width)
                    };
                    let sample = DeviationSampleRef {
                        deviation: unsafe { run_sym.get_unchecked(..self.num_deviation_bits) },
                        id: unsafe { run_sym.get_unchecked(self.num_deviation_bits..) },
                    };
                    for _ in 0..(emit_end - emit_start) {
                        f(DeviationSampleRef {
                            deviation: sample.deviation,
                            id: sample.id,
                        })?;
                        yielded += 1;
                    }
                }

                processed += run_len;
                sym_cursor += symbol_width;
            }

            if yielded >= capped_count {
                break;
            }

            let literals = m_count as usize;
            let lit_start = processed;
            let lit_end = processed + literals;
            let emit_start = lit_start.max(skip_from_boundary);
            let emit_end = lit_end.min(skip_from_boundary + capped_count);

            if emit_start < emit_end {
                let start_offset_in_lit = emit_start - lit_start;
                let start_bit = sym_cursor + start_offset_in_lit * symbol_width;

                for i in 0..(emit_end - emit_start) {
                    let s = start_bit + i * symbol_width;
                    if s + symbol_width > self.symbol_bit_stream.len() {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "RLE offset literal exceeds symbol stream length".to_string(),
                        });
                    }
                    let sym = unsafe { self.symbol_bit_stream.get_unchecked(s..s + symbol_width) };
                    f(DeviationSampleRef {
                        deviation: unsafe { sym.get_unchecked(..self.num_deviation_bits) },
                        id: unsafe { sym.get_unchecked(self.num_deviation_bits..) },
                    })?;
                    yielded += 1;
                }
            }

            processed += literals;
            sym_cursor += literals * symbol_width;
        }

        Ok(())
    }

    pub fn to_deviation_data(&self) -> Result<DeviationData, EntroGdError> {
        let _timer =
            ScopedTimer::debug("Converting RLE offset deviation data to raw deviation data");
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let expected_raw_bits = self.num_samples.checked_mul(symbol_width).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "raw deviation stream length overflow".to_string(),
            }
        })?;

        let mut raw = BitVec::with_capacity(expected_raw_bits);
        let mut symbol_cursor = 0usize;
        let mut decoded_samples = 0usize;

        for &(r_encoded, m_count) in &self.rm_values {
            if r_encoded > 0 {
                let run_len = (r_encoded as usize) + 1;
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE offset run symbol exceeds symbol stream length".to_string(),
                    });
                }

                let run_symbol = unsafe {
                    self.symbol_bit_stream
                        .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                };
                for _ in 0..run_len {
                    raw.extend_from_bitslice(run_symbol);
                }
                symbol_cursor += symbol_width;
                decoded_samples += run_len;
            }

            let literal_count = m_count as usize;
            for _ in 0..literal_count {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE offset literal symbol exceeds symbol stream length"
                            .to_string(),
                    });
                }

                raw.extend_from_bitslice(unsafe {
                    self.symbol_bit_stream
                        .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                });
                symbol_cursor += symbol_width;
            }
            decoded_samples += literal_count;
        }

        if decoded_samples != self.num_samples {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE offset decoded sample count mismatch: expected {}, got {}",
                    self.num_samples, decoded_samples
                ),
            });
        }

        if symbol_cursor != self.symbol_bit_stream.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE offset symbol stream not fully consumed: consumed {} of {} bits",
                    symbol_cursor,
                    self.symbol_bit_stream.len()
                ),
            });
        }

        Ok(DeviationData::new(
            raw,
            self.num_samples,
            self.num_deviation_bits,
            self.num_id_bits,
        ))
    }

    fn decode_sample_from(
        &self,
        start_rm_idx: usize,
        start_symbol_bit: usize,
        sample_offset: usize,
    ) -> Option<DeviationSample> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let mut symbol_cursor = start_symbol_bit;
        let mut logical_idx = 0usize;

        for &(r_encoded, m_count) in self.rm_values.iter().skip(start_rm_idx) {
            if r_encoded > 0 {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return None;
                }
                let run_symbol = unsafe {
                    self.symbol_bit_stream
                        .get_unchecked(symbol_cursor..symbol_cursor + symbol_width)
                }
                .to_bitvec();
                symbol_cursor += symbol_width;

                let run_len = (r_encoded as usize) + 1;
                if sample_offset < logical_idx + run_len {
                    return Some(DeviationSample {
                        deviation: run_symbol[0..self.num_deviation_bits].to_bitvec(),
                        id: run_symbol[self.num_deviation_bits..].to_bitvec(),
                    });
                }
                logical_idx += run_len;
            }

            let literals = m_count as usize;
            if sample_offset < logical_idx + literals {
                let offset = sample_offset - logical_idx;
                let start = symbol_cursor + offset * symbol_width;
                let end = start + symbol_width;
                if end > self.symbol_bit_stream.len() {
                    return None;
                }
                let symbol =
                    unsafe { self.symbol_bit_stream.get_unchecked(start..end) }.to_bitvec();
                return Some(DeviationSample {
                    deviation: symbol[0..self.num_deviation_bits].to_bitvec(),
                    id: symbol[self.num_deviation_bits..].to_bitvec(),
                });
            }

            symbol_cursor += literals * symbol_width;
            logical_idx += literals;
        }

        None
    }
}

fn append_u32_bits(out: &mut BitVec<usize, Lsb0>, value: u32) {
    for shift in 0..32 {
        out.push(((value >> shift) & 1) == 1);
    }
}
