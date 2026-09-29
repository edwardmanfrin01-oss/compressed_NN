use fxhash::FxHashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::path_utils::ensure_egd_extension;
use super::tags::{
    BASE_TABLE_TAG_DELTA_FIXED, BASE_TABLE_TAG_DELTA_UNARY, BASE_TABLE_TAG_RAW,
    ENCODING_TAG_HUFFMAN_BASE_ID_ONLY, ENCODING_TAG_NORMAL, ENCODING_TAG_RLE_OFFSET,
    decode_data_type, encode_data_type,
};

use crate::ScopedTimer;
use crate::compression::base_table::{BaseBitLayoutState, BaseLayoutInfo};
use crate::compression::data::{
    BitDataInfo, BitDataSet, FeatureDataType, FeatureSpec, FeatureTransform,
};
use crate::compression::decompression::decompress_file;
use crate::compression::encoding::{
    BaseTable, CompressedData, DeltaBaseTableData, DeviationData, EncodedData,
    HuffmanDeviationData, RLE_LONG_MAX, RleDeviationOffsetData,
};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::utils::bits_needed_nonzero;
use bitvec::prelude::*;

pub const MAGIC_BYTES: [u8; 3] = *b"EGD";
pub const FORMAT_VERSION: u8 = 1;

fn rle_symbol_count(rm_values: &[(u8, u8)]) -> Result<usize, EntroGdError> {
    rm_values
        .iter()
        .try_fold(0usize, |acc, (r, m_val)| {
            let run_symbols = if *r > 0 { 1usize } else { 0usize };
            let literals = *m_val as usize;
            acc.checked_add(run_symbols + literals)
        })
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "rm symbol count overflow".to_string(),
        })
}

fn write_int_le(buf: &mut [u8], value: u64, num_bytes: usize) {
    let le = value.to_le_bytes();
    buf[..num_bytes].copy_from_slice(&le[..num_bytes]);
}

fn read_int_le(buf: &[u8], num_bytes: usize) -> u64 {
    let mut le = [0u8; 8];
    le[..num_bytes].copy_from_slice(&buf[..num_bytes]);
    u64::from_le_bytes(le)
}

fn write_bitvec_as_bytes(out: &mut [u8], bv: &BitVec<usize, Lsb0>, bit_len: usize) {
    let byte_len = bit_len.div_ceil(8);
    let word_size = std::mem::size_of::<usize>();
    let words = bv.as_raw_slice();
    for (word_idx, &word) in words.iter().enumerate() {
        let start = word_idx * word_size;
        if start >= byte_len {
            break;
        }
        let end = (start + word_size).min(byte_len);
        let copy_len = end - start;
        out[start..end].copy_from_slice(&word.to_ne_bytes()[..copy_len]);
    }
    if !bit_len.is_multiple_of(8) && byte_len > 0 {
        out[byte_len - 1] &= (1u8 << (bit_len % 8)) - 1;
    }
    if bit_len == 0 && byte_len > 0 {
        out[0] = 0;
    }
}

fn read_bitvec_from_bytes(bytes: &[u8], bit_len: usize) -> BitVec<usize, Lsb0> {
    if bit_len == 0 {
        return BitVec::new();
    }
    let byte_len = bit_len.div_ceil(8);
    let word_size = std::mem::size_of::<usize>();
    let word_count = byte_len.div_ceil(word_size);
    let mut words = vec![0usize; word_count];
    for (word_idx, word) in words.iter_mut().enumerate() {
        let start = word_idx * word_size;
        let end = (start + word_size).min(byte_len);
        let mut buf = [0u8; std::mem::size_of::<usize>()];
        buf[..end - start].copy_from_slice(&bytes[start..end]);
        *word = usize::from_ne_bytes(buf);
    }
    let mut bv = BitVec::from_vec(words);
    bv.truncate(bit_len);
    bv
}

/// In-memory EGD file contents that can be saved to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgdFile {
    bytes: Vec<u8>,
}

impl EgdFile {
    pub fn from_compressed_data(compressed: CompressedData) -> Result<Self, EntroGdError> {
        let data_info = &compressed.metadata;
        let num_features = data_info.num_features();
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        let chunk_size = data_info.chunk_size();
        let original_num_rows = data_info.original_size_bits() / chunk_size;
        let n = original_num_rows;
        let condensed_weights = compressed
            .condensed_sample_weights
            .as_deref()
            .unwrap_or(&[]);
        let m = condensed_weights.len();

        let num_bases = compressed.base_table.len();
        let num_id_bits = bits_needed_nonzero(num_bases);
        let base_bit_positions = compressed.layout.selected_base_bit_positions();
        let num_deviation_bits = chunk_size.saturating_sub(base_bit_positions.len());
        let bits_per_sample = num_deviation_bits + num_id_bits;
        let _num_samples = n
            .checked_add(m)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "n + m overflows usize".to_string(),
            })?;

        // ---- Pre-compute all section sizes ----

        let header_size: usize = 3 + 1 + 8 + 8 + 8; // 28

        let mut feature_sizes = Vec::with_capacity(num_features);
        let mut total_feature_size = 0usize;
        for feature_idx in 0..num_features {
            let spec = data_info.feature_spec(feature_idx);
            let bits = data_info.feature_bits(feature_idx);
            let transform_size = match spec.transform {
                FeatureTransform::None => 0,
                FeatureTransform::ScaledSignedInt { .. } => 1,
                FeatureTransform::OffsetSignedInt { .. } => 8,
                FeatureTransform::OffsetUnsignedInt { .. } => 8,
                FeatureTransform::ScaledOffsetSignedInt { .. } => 9,
            };
            let bit_states_bytes = (2 * bits).div_ceil(8);
            let size = 3 + transform_size + 2 + bit_states_bytes;
            feature_sizes.push(size);
            total_feature_size += size;
        }

        let weight_bits = bits_needed_nonzero(n);
        let weight_bytes = weight_bits.div_ceil(8);
        let weights_size = m * weight_bytes;

        let variable_positions = compressed.layout.variable_base_bit_positions();
        let v = variable_positions.len();

        let base_table_size = match &compressed.base_table {
            BaseTable::Raw(rows) => {
                let base_entry_bytes = v.div_ceil(8);
                1 + 8 + rows.len() * base_entry_bytes
            }
            BaseTable::Delta(delta) => {
                let lb = variable_positions.len();
                let index_bits = bits_needed_nonzero(lb.max(1));
                let index_bytes = index_bits.div_ceil(8);
                let order_len = delta.sort_column_order.len();
                let fsk_bytes = if delta.num_bases > 0 {
                    lb.div_ceil(8)
                } else {
                    0
                };
                let delta_byte_len = delta.delta_bit_stream.len().div_ceil(8);
                1 + 8 + 8 + order_len * index_bytes + fsk_bytes + 8 + 8 + delta_byte_len
            }
        };

        let encoded_data_size = match &compressed.encoded_data {
            EncodedData::Normal(raw) => {
                let stream_bit_len = raw.get_num_samples() * bits_per_sample;
                1 + stream_bit_len.div_ceil(8)
            }
            EncodedData::RleOffset(rle) => {
                let row_count = rle.row_offsets().len();
                let rm_count = rle.rm_values().len();
                let symbol_count = rle_symbol_count(rle.rm_values())?;
                let sym_byte_len = (symbol_count * bits_per_sample).div_ceil(8);
                1 + 4 + 4 + row_count * 8 + 4 + rm_count * 2 + sym_byte_len
            }
            EncodedData::Huffman(huff) => {
                let symbol_count = huff.canonical_symbols().len();
                let l_id = huff.get_num_id_bits();
                let l_d = huff.get_num_deviation_bits();
                let row_count = huff.row_offsets().len();
                let symbol_width = l_id;
                let symbol_bytes = symbol_width.div_ceil(8);
                let canonical_bytes = symbol_count * (symbol_bytes + 1);
                let row_offsets_bytes = row_count * 4;
                let dev_byte_len = (huff.get_num_samples() * l_d).div_ceil(8);
                let pixel_bit_len = huff.pixel_bit_stream().len();
                let pixel_byte_len = if pixel_bit_len == 0 {
                    0
                } else {
                    pixel_bit_len.div_ceil(8)
                };
                1 + 4
                    + 1
                    + 1
                    + 4
                    + 4
                    + canonical_bytes
                    + row_offsets_bytes
                    + dev_byte_len
                    + pixel_byte_len
            }
        };

        let total =
            header_size + total_feature_size + weights_size + base_table_size + encoded_data_size;
        let mut buf = vec![0u8; total];
        let mut off = 0usize;

        // ---- Section 1: Header ----
        buf[off..off + 3].copy_from_slice(&MAGIC_BYTES);
        off += 3;
        buf[off] = FORMAT_VERSION;
        off += 1;
        buf[off..off + 8].copy_from_slice(&(n as u64).to_le_bytes());
        off += 8;
        buf[off..off + 8].copy_from_slice(&(m as u64).to_le_bytes());
        off += 8;
        buf[off..off + 8].copy_from_slice(&(num_features as u64).to_le_bytes());
        off += 8;

        // ---- Section 2: Feature metadata ----
        let mut variable_position_to_index =
            FxHashMap::with_capacity_and_hasher(v, Default::default());
        for (idx, &bit_pos) in variable_positions.iter().enumerate() {
            variable_position_to_index.insert(bit_pos, idx);
        }

        for feature_idx in 0..num_features {
            let spec = data_info.feature_spec(feature_idx);
            let bits = data_info.feature_bits(feature_idx);

            buf[off] = 1;
            off += 1; // tag
            buf[off] = encode_data_type(spec.data_type);
            off += 1;

            match spec.transform {
                FeatureTransform::None => {
                    buf[off] = 0;
                    off += 1;
                }
                FeatureTransform::ScaledSignedInt { decimal_scale } => {
                    buf[off] = 1;
                    off += 1;
                    buf[off] = decimal_scale;
                    off += 1;
                }
                FeatureTransform::OffsetSignedInt { min_value } => {
                    buf[off] = 2;
                    off += 1;
                    buf[off..off + 8].copy_from_slice(&(min_value as u64).to_le_bytes());
                    off += 8;
                }
                FeatureTransform::OffsetUnsignedInt { min_value } => {
                    buf[off] = 3;
                    off += 1;
                    buf[off..off + 8].copy_from_slice(&min_value.to_le_bytes());
                    off += 8;
                }
                FeatureTransform::ScaledOffsetSignedInt {
                    decimal_scale,
                    min_value,
                } => {
                    buf[off] = 4;
                    off += 1;
                    buf[off] = decimal_scale;
                    off += 1;
                    buf[off..off + 8].copy_from_slice(&(min_value as u64).to_le_bytes());
                    off += 8;
                }
            }

            buf[off..off + 2].copy_from_slice(&(bits as u16).to_le_bytes());
            off += 2;

            let offset = data_info.feature_offset(feature_idx);
            for local_bit in 0..bits {
                let global_bit = offset + local_bit;
                let state: u8 = match compressed.layout.state_at(global_bit) {
                    BaseBitLayoutState::Deviation => 0,
                    BaseBitLayoutState::Variable => 1,
                    BaseBitLayoutState::ConstantZero => 2,
                    BaseBitLayoutState::ConstantOne => 3,
                };
                let byte_idx = local_bit / 4;
                let bit_shift = ((local_bit % 4) * 2) as u32;
                buf[off + byte_idx] |= state << bit_shift;
            }
            off += (2 * bits).div_ceil(8);
        }

        // ---- Section 3: Condensed sample weights ----
        for &weight in condensed_weights {
            write_int_le(
                &mut buf[off..off + weight_bytes],
                weight as u64,
                weight_bytes,
            );
            off += weight_bytes;
        }

        // ---- Section 4: Base table ----
        match &compressed.base_table {
            BaseTable::Raw(rows) => {
                buf[off] = BASE_TABLE_TAG_RAW;
                off += 1;
                buf[off..off + 8].copy_from_slice(&(rows.len() as u64).to_le_bytes());
                off += 8;

                let base_entry_bytes = v.div_ceil(8);
                for (base_bits, _) in rows {
                    let entry_start = off;
                    for b in &mut buf[entry_start..entry_start + base_entry_bytes] {
                        *b = 0;
                    }
                    for (local_idx, &global_bit) in variable_positions.iter().enumerate() {
                        let variable_idx = *variable_position_to_index
                            .get(&global_bit)
                            .ok_or_else(|| EntroGdError::InvalidMetadata {
                                message: format!(
                                    "variable base position {} missing from index map",
                                    global_bit
                                ),
                            })?;
                        if base_bits.get(variable_idx).map(|r| *r).unwrap_or(false) {
                            buf[entry_start + local_idx / 8] |= 1u8 << (local_idx % 8);
                        }
                    }
                    off += base_entry_bytes;
                }
            }
            BaseTable::Delta(delta) => {
                buf[off] = delta.codec_id;
                off += 1;
                buf[off..off + 8].copy_from_slice(&(delta.num_bases as u64).to_le_bytes());
                off += 8;

                let lb = variable_positions.len();
                let index_bits = bits_needed_nonzero(lb.max(1));
                let index_bytes = index_bits.div_ceil(8);
                let order_len = delta.sort_column_order.len();
                buf[off..off + 8].copy_from_slice(&(order_len as u64).to_le_bytes());
                off += 8;

                for &current_idx in &delta.sort_column_order {
                    let &global_bit = variable_positions.get(current_idx).ok_or_else(|| {
                        EntroGdError::InvalidMetadata {
                            message: format!(
                                "delta sort column index {} out of range {}",
                                current_idx,
                                variable_positions.len()
                            ),
                        }
                    })?;
                    let meta_idx =
                        *variable_position_to_index.get(&global_bit).ok_or_else(|| {
                            EntroGdError::InvalidMetadata {
                                message: format!(
                                    "variable base position {} missing from index map",
                                    global_bit
                                ),
                            }
                        })?;
                    write_int_le(
                        &mut buf[off..off + index_bytes],
                        meta_idx as u64,
                        index_bytes,
                    );
                    off += index_bytes;
                }

                if delta.num_bases > 0 {
                    let fsk_bytes = lb.div_ceil(8);
                    for b in &mut buf[off..off + fsk_bytes] {
                        *b = 0;
                    }
                    for idx in 0..lb {
                        if delta.first_sort_key.get(idx).map(|r| *r).unwrap_or(false) {
                            buf[off + idx / 8] |= 1u8 << (idx % 8);
                        }
                    }
                    off += fsk_bytes;
                }

                buf[off..off + 8].copy_from_slice(&(delta.delta_count as u64).to_le_bytes());
                off += 8;
                let delta_bit_len = delta.delta_bit_stream.len();
                buf[off..off + 8].copy_from_slice(&(delta_bit_len as u64).to_le_bytes());
                off += 8;
                let delta_byte_len = delta_bit_len.div_ceil(8);
                write_bitvec_as_bytes(
                    &mut buf[off..off + delta_byte_len],
                    &delta.delta_bit_stream,
                    delta_bit_len,
                );
                off += delta_byte_len;
            }
        }

        // ---- Section 5: Encoded data ----
        match &compressed.encoded_data {
            EncodedData::Normal(raw) => {
                buf[off] = ENCODING_TAG_NORMAL;
                off += 1;
                let stream_bit_len = raw.get_num_samples() * bits_per_sample;
                let stream_byte_len = stream_bit_len.div_ceil(8);
                write_bitvec_as_bytes(
                    &mut buf[off..off + stream_byte_len],
                    raw.encoded_bit_stream(),
                    stream_bit_len,
                );
                off += stream_byte_len;
            }
            EncodedData::RleOffset(rle) => {
                buf[off] = ENCODING_TAG_RLE_OFFSET;
                off += 1;
                let row_count = rle.row_offsets().len();
                buf[off..off + 4].copy_from_slice(&(row_count as u32).to_le_bytes());
                off += 4;
                buf[off..off + 4].copy_from_slice(&(rle.row_width() as u32).to_le_bytes());
                off += 4;

                for &(rm_idx, symbol_bit_idx) in rle.row_offsets() {
                    buf[off..off + 4].copy_from_slice(&rm_idx.to_le_bytes());
                    off += 4;
                    buf[off..off + 4].copy_from_slice(&symbol_bit_idx.to_le_bytes());
                    off += 4;
                }

                let rm_count = rle.rm_values().len();
                buf[off..off + 4].copy_from_slice(&(rm_count as u32).to_le_bytes());
                off += 4;
                for &(r, m_val) in rle.rm_values() {
                    buf[off] = r;
                    off += 1;
                    buf[off] = m_val;
                    off += 1;
                }

                let symbol_count = rle_symbol_count(rle.rm_values())?;
                let sym_bit_len = symbol_count * bits_per_sample;
                let sym_byte_len = sym_bit_len.div_ceil(8);
                write_bitvec_as_bytes(
                    &mut buf[off..off + sym_byte_len],
                    rle.symbol_bit_stream(),
                    sym_bit_len,
                );
                off += sym_byte_len;
            }
            EncodedData::Huffman(huff) => {
                buf[off] = ENCODING_TAG_HUFFMAN_BASE_ID_ONLY;
                off += 1;

                let symbol_count = huff.canonical_symbols().len();
                let l_id = huff.get_num_id_bits();
                let l_d = huff.get_num_deviation_bits();
                let row_count = huff.row_offsets().len();

                buf[off..off + 4].copy_from_slice(&(symbol_count as u32).to_le_bytes());
                off += 4;
                buf[off] = l_id as u8;
                off += 1;
                buf[off] = l_d as u8;
                off += 1;
                buf[off..off + 4].copy_from_slice(&(row_count as u32).to_le_bytes());
                off += 4;
                buf[off..off + 4].copy_from_slice(&(huff.row_width() as u32).to_le_bytes());
                off += 4;

                let symbol_bytes = l_id.div_ceil(8);
                for (&symbol, &code_len) in huff
                    .canonical_symbols()
                    .iter()
                    .zip(huff.canonical_code_lengths().iter())
                {
                    write_int_le(&mut buf[off..off + symbol_bytes], symbol, symbol_bytes);
                    off += symbol_bytes;
                    buf[off] = code_len;
                    off += 1;
                }

                for &offset in huff.row_offsets() {
                    buf[off..off + 4].copy_from_slice(&offset.to_le_bytes());
                    off += 4;
                }

                let dev_bit_len = huff.get_num_samples() * l_d;
                let dev_byte_len = dev_bit_len.div_ceil(8);
                write_bitvec_as_bytes(
                    &mut buf[off..off + dev_byte_len],
                    huff.raw_deviation_bit_stream(),
                    dev_bit_len,
                );
                off += dev_byte_len;

                let pixel_bit_len = huff.pixel_bit_stream().len();
                let pixel_byte_len = if pixel_bit_len == 0 {
                    0
                } else {
                    pixel_bit_len.div_ceil(8)
                };
                write_bitvec_as_bytes(
                    &mut buf[off..off + pixel_byte_len],
                    huff.pixel_bit_stream(),
                    pixel_bit_len,
                );
                off += pixel_byte_len;
            }
        }

        assert_eq!(off, total, "buffer write overflow/underflow");

        Ok(EgdFile { bytes: buf })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        EgdFile { bytes }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EntroGdError> {
        let bytes = fs::read(path)?;
        Ok(EgdFile { bytes })
    }

    pub fn to_compressed_data(&self) -> Result<CompressedData, EntroGdError> {
        let bytes = &self.bytes;
        let mut off = 0usize;

        // ---- Section 1: Header ----
        if bytes.len() < 28 {
            return Err(EntroGdError::InvalidMetadata {
                message: "EGD file too short for header".to_string(),
            });
        }
        if bytes[0..3] != MAGIC_BYTES {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid magic bytes (expected EGD)".to_string(),
            });
        }
        off += 3;
        let version = bytes[off];
        off += 1;
        if version != FORMAT_VERSION {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported format version {} (expected {})",
                    version, FORMAT_VERSION
                ),
            });
        }

        let n = usize::try_from(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()))
            .map_err(|_| EntroGdError::InvalidMetadata {
                message: "n does not fit into usize".to_string(),
            })?;
        off += 8;
        let m = usize::try_from(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()))
            .map_err(|_| EntroGdError::InvalidMetadata {
                message: "m does not fit into usize".to_string(),
            })?;
        off += 8;
        let num_features = usize::try_from(u64::from_le_bytes(
            bytes[off..off + 8].try_into().unwrap(),
        ))
        .map_err(|_| EntroGdError::InvalidMetadata {
            message: "num_features does not fit into usize".to_string(),
        })?;
        off += 8;
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        // ---- Section 2: Feature metadata ----
        let mut features = Vec::with_capacity(num_features);
        let mut base_bit_positions = Vec::new();
        let mut variable_base_bit_positions = Vec::new();
        let mut bit_states = Vec::new();
        let mut running_offset = 0usize;

        for feature_idx in 0..num_features {
            if off >= bytes.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "unexpected end of EGD while reading feature {} metadata",
                        feature_idx
                    ),
                });
            }

            let tag = bytes[off];
            off += 1;
            if tag != 1 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "unsupported feature metadata tag {} at index {}",
                        tag, feature_idx
                    ),
                });
            }

            let data_type = decode_data_type(bytes[off])?;
            off += 1;

            let transform = match bytes[off] {
                0 => {
                    off += 1;
                    FeatureTransform::None
                }
                1 => {
                    off += 1;
                    let decimal_scale = bytes[off];
                    off += 1;
                    FeatureTransform::ScaledSignedInt { decimal_scale }
                }
                2 => {
                    off += 1;
                    let min_value = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                    off += 8;
                    FeatureTransform::OffsetSignedInt { min_value }
                }
                3 => {
                    off += 1;
                    let min_value = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                    off += 8;
                    FeatureTransform::OffsetUnsignedInt { min_value }
                }
                4 => {
                    off += 1;
                    let decimal_scale = bytes[off];
                    off += 1;
                    let min_value = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                    off += 8;
                    FeatureTransform::ScaledOffsetSignedInt {
                        decimal_scale,
                        min_value,
                    }
                }
                transform_tag => {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "unsupported transform tag {} at feature {}",
                            transform_tag, feature_idx
                        ),
                    });
                }
            };

            let bits = usize::from(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()));
            off += 2;
            if bits == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("feature {} has 0 bits", feature_idx),
                });
            }
            let bit_states_bytes = (2 * bits).div_ceil(8);
            if off + bit_states_bytes > bytes.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "unexpected end of EGD while reading bit states for feature {}",
                        feature_idx
                    ),
                });
            }

            let data_type = if data_type == FeatureDataType::UInt(0) {
                FeatureDataType::UInt(bits as u16)
            } else {
                data_type
            };

            features.push(FeatureSpec {
                data_type,
                transform,
            });

            for local_bit in 0..bits {
                let global_bit = running_offset + local_bit;
                let state = ((bytes[off + local_bit / 4] >> ((local_bit % 4) * 2)) & 0b11) as usize;
                match state {
                    0 => {
                        bit_states.push(BaseBitLayoutState::Deviation);
                    }
                    1 => {
                        bit_states.push(BaseBitLayoutState::Variable);
                        base_bit_positions.push(global_bit);
                        variable_base_bit_positions.push(global_bit);
                    }
                    2 => {
                        bit_states.push(BaseBitLayoutState::ConstantZero);
                        base_bit_positions.push(global_bit);
                    }
                    3 => {
                        bit_states.push(BaseBitLayoutState::ConstantOne);
                        base_bit_positions.push(global_bit);
                    }
                    _ => {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid base bit-state {} at feature {} local bit {}",
                                state, feature_idx, local_bit
                            ),
                        });
                    }
                }
            }
            off += bit_states_bytes;
            running_offset =
                running_offset
                    .checked_add(bits)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "chunk size overflow while parsing features".to_string(),
                    })?;
        }

        // ---- Section 3: Condensed sample weights ----
        let weight_bits = bits_needed_nonzero(n);
        let weight_bytes = weight_bits.div_ceil(8);
        let weights_size = m * weight_bytes;
        if off + weights_size > bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "unexpected end of EGD while reading condensed weights".to_string(),
            });
        }
        let mut weights = Vec::with_capacity(m);
        for _ in 0..m {
            let val = read_int_le(&bytes[off..off + weight_bytes], weight_bytes) as usize;
            weights.push(val);
            off += weight_bytes;
        }

        // ---- Section 4: Base table ----
        let _timer = ScopedTimer::info("Loading base table");
        if off >= bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "unexpected end of EGD before base table".to_string(),
            });
        }
        let base_table_tag = bytes[off];
        off += 1;
        let num_bases = usize::try_from(u64::from_le_bytes(
            bytes[off..off + 8].try_into().unwrap(),
        ))
        .map_err(|_| EntroGdError::InvalidMetadata {
            message: "num_bases does not fit into usize".to_string(),
        })?;
        off += 8;

        let chunk_size = features.iter().map(|f| f.data_type.bits()).sum::<usize>();
        if base_bit_positions.len() > chunk_size {
            return Err(EntroGdError::InvalidMetadata {
                message: "base bit positions exceed chunk size".to_string(),
            });
        }

        let mut entropy_sorted_column_order: Option<Vec<usize>> = None;
        let mut base_table = match base_table_tag {
            BASE_TABLE_TAG_RAW => {
                let v = variable_base_bit_positions.len();
                let base_entry_bytes = v.div_ceil(8);
                let section_len = num_bases * base_entry_bytes;
                if off + section_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading base table".to_string(),
                    });
                }
                let mut rows = Vec::with_capacity(num_bases);
                for base_idx in 0..num_bases {
                    let entry_start = off + base_idx * base_entry_bytes;
                    let mut base_bits = BitVec::with_capacity(v);
                    for local_bit in 0..v {
                        let bit = (bytes[entry_start + local_bit / 8] >> (local_bit % 8)) & 1;
                        base_bits.push(bit != 0);
                    }
                    rows.push((base_bits, 0usize));
                }
                off += section_len;
                BaseTable::Raw(rows)
            }
            BASE_TABLE_TAG_DELTA_UNARY | BASE_TABLE_TAG_DELTA_FIXED => {
                let order_len =
                    usize::try_from(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "sort_column_order length does not fit into usize".to_string(),
                        })?;
                off += 8;

                let lb = variable_base_bit_positions.len();
                let index_bits = bits_needed_nonzero(lb.max(1));
                let index_bytes = index_bits.div_ceil(8);
                let order_section_len = order_len * index_bytes;
                if off + order_section_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading delta order".to_string(),
                    });
                }
                let mut order = Vec::with_capacity(order_len);
                for _ in 0..order_len {
                    let idx = read_int_le(&bytes[off..off + index_bytes], index_bytes) as usize;
                    off += index_bytes;
                    if idx >= lb {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!("delta sort column index {} out of range {}", idx, lb),
                        });
                    }
                    order.push(idx);
                }

                let mut first_sort_key = BitVec::with_capacity(lb);
                if num_bases > 0 {
                    let fsk_bytes = lb.div_ceil(8);
                    if off + fsk_bytes > bytes.len() {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "unexpected end of EGD while reading delta first sort key"
                                .to_string(),
                        });
                    }
                    for idx in 0..lb {
                        let bit = (bytes[off + idx / 8] >> (idx % 8)) & 1;
                        first_sort_key.push(bit != 0);
                    }
                    off += fsk_bytes;
                }

                let delta_count =
                    usize::try_from(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "delta_count does not fit into usize".to_string(),
                        })?;
                off += 8;

                let delta_bit_len =
                    usize::try_from(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "delta bitstream length does not fit into usize".to_string(),
                        })?;
                off += 8;

                let delta_byte_len = delta_bit_len.div_ceil(8);
                if off + delta_byte_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading delta bitstream".to_string(),
                    });
                }
                let delta_bit_stream =
                    read_bitvec_from_bytes(&bytes[off..off + delta_byte_len], delta_bit_len);
                off += delta_byte_len;

                entropy_sorted_column_order = Some(order.clone());
                BaseTable::Delta(DeltaBaseTableData {
                    num_bases,
                    first_sort_key,
                    delta_bit_stream,
                    delta_count,
                    sort_column_order: order,
                    codec_id: base_table_tag,
                })
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unsupported base-table tag {}", other),
                });
            }
        };
        drop(_timer);

        let num_samples = n
            .checked_add(m)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "n + m overflows usize".to_string(),
            })?;
        let num_id_bits = bits_needed_nonzero(num_bases);
        let num_deviation_bits = chunk_size.saturating_sub(base_bit_positions.len());
        let bits_per_sample = num_deviation_bits + num_id_bits;

        // ---- Section 5: Encoded data ----
        if off >= bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "unexpected end of EGD before encoded data".to_string(),
            });
        }
        let encoding_tag = bytes[off];
        off += 1;
        let encoded_data = match encoding_tag {
            ENCODING_TAG_NORMAL => {
                let expected_bit_len = num_samples * bits_per_sample;
                let expected_byte_len = expected_bit_len.div_ceil(8);
                if off + expected_byte_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading normal encoded stream"
                            .to_string(),
                    });
                }
                let encoded_stream =
                    read_bitvec_from_bytes(&bytes[off..off + expected_byte_len], expected_bit_len);
                off += expected_byte_len;
                EncodedData::Normal(DeviationData::new(
                    encoded_stream,
                    num_samples,
                    num_deviation_bits,
                    num_id_bits,
                ))
            }
            ENCODING_TAG_RLE_OFFSET => {
                let row_count =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "RLE row count does not fit into usize".to_string(),
                        })?;
                off += 4;
                let row_width =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "RLE row width does not fit into usize".to_string(),
                        })?;
                off += 4;

                let row_offsets_section = row_count * 8;
                if off + row_offsets_section > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading RLE row offsets".to_string(),
                    });
                }
                let mut row_offsets = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    let rm_idx = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    let symbol_bit_idx =
                        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    row_offsets.push((rm_idx, symbol_bit_idx));
                }

                let rm_count =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "RLE rm pair count does not fit into usize".to_string(),
                        })?;
                off += 4;
                let rm_section = rm_count * 2;
                if off + rm_section > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading RLE rm values".to_string(),
                    });
                }
                let mut rm_values = Vec::with_capacity(rm_count);
                for _ in 0..rm_count {
                    let r = bytes[off];
                    off += 1;
                    let m_val = bytes[off];
                    off += 1;
                    if r > RLE_LONG_MAX {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "invalid RLE r value".to_string(),
                        });
                    }
                    if m_val > RLE_LONG_MAX {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "invalid RLE m_val value".to_string(),
                        });
                    }
                    rm_values.push((r, m_val));
                }

                let symbol_count = rle_symbol_count(&rm_values)?;
                let sym_bit_len = symbol_count * bits_per_sample;
                let sym_byte_len = sym_bit_len.div_ceil(8);
                if off + sym_byte_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading RLE symbol stream"
                            .to_string(),
                    });
                }
                let symbol_stream =
                    read_bitvec_from_bytes(&bytes[off..off + sym_byte_len], sym_bit_len);
                off += sym_byte_len;

                let rle = RleDeviationOffsetData::new(
                    symbol_stream,
                    rm_values,
                    row_offsets,
                    n,
                    row_width,
                    num_samples,
                    num_deviation_bits,
                    num_id_bits,
                )?;

                if row_count != rle.row_offsets().len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "RLE row offset count mismatch: header={}, decoded={}",
                            row_count,
                            rle.row_offsets().len()
                        ),
                    });
                }

                EncodedData::RleOffset(rle)
            }
            ENCODING_TAG_HUFFMAN_BASE_ID_ONLY => {
                let symbol_count =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "Huffman symbol count does not fit into usize".to_string(),
                        })?;
                off += 4;
                let huffman_num_id_bits = usize::from(bytes[off]);
                off += 1;
                let huffman_num_deviation_bits = usize::from(bytes[off]);
                off += 1;
                let row_count =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "Huffman row count does not fit into usize".to_string(),
                        })?;
                off += 4;
                let row_width =
                    usize::try_from(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
                        .map_err(|_| EntroGdError::InvalidMetadata {
                            message: "Huffman row width does not fit into usize".to_string(),
                        })?;
                off += 4;

                if huffman_num_id_bits != num_id_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_id mismatch: header={}, expected={}",
                            huffman_num_id_bits, num_id_bits
                        ),
                    });
                }
                if huffman_num_deviation_bits != num_deviation_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_d mismatch: header={}, expected={}",
                            huffman_num_deviation_bits, num_deviation_bits
                        ),
                    });
                }

                let symbol_width = huffman_num_id_bits;
                let symbol_bytes = symbol_width.div_ceil(8);

                let canonical_section = symbol_count * (symbol_bytes + 1);
                if off + canonical_section > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading Huffman canonical table"
                            .to_string(),
                    });
                }
                let mut canonical_symbols = Vec::with_capacity(symbol_count);
                let mut canonical_code_lengths = Vec::with_capacity(symbol_count);
                for _ in 0..symbol_count {
                    let symbol = read_int_le(&bytes[off..off + symbol_bytes], symbol_bytes);
                    off += symbol_bytes;
                    canonical_symbols.push(symbol);
                    canonical_code_lengths.push(bytes[off]);
                    off += 1;
                }

                let row_offsets_section = row_count * 4;
                if off + row_offsets_section > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading Huffman row offsets"
                            .to_string(),
                    });
                }
                let mut row_offsets = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    row_offsets.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                    off += 4;
                }

                let dev_bit_len = num_samples * huffman_num_deviation_bits;
                let dev_byte_len = dev_bit_len.div_ceil(8);
                if off + dev_byte_len > bytes.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "unexpected end of EGD while reading Huffman raw deviation stream"
                            .to_string(),
                    });
                }
                let raw_deviation_stream =
                    read_bitvec_from_bytes(&bytes[off..off + dev_byte_len], dev_bit_len);
                off += dev_byte_len;

                let pixel_bit_stream =
                    read_bitvec_from_bytes(&bytes[off..], (bytes.len() - off) * 8);
                off = bytes.len();

                EncodedData::Huffman(HuffmanDeviationData::new(
                    pixel_bit_stream,
                    raw_deviation_stream,
                    canonical_symbols,
                    canonical_code_lengths,
                    row_offsets,
                    num_samples,
                    n,
                    huffman_num_deviation_bits,
                    huffman_num_id_bits,
                    row_width,
                )?)
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unsupported encoded-data tag {}", other),
                });
            }
        };

        // Verify trailing bytes are zero (for non-Huffman, which consumes rest)
        if encoding_tag != ENCODING_TAG_HUFFMAN_BASE_ID_ONLY {
            for &b in &bytes[off..] {
                if b != 0 {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "non-zero trailing bytes after encoded stream".to_string(),
                    });
                }
            }
        }

        // Reconstruct base frequencies from encoded IDs
        if num_bases > 0
            && let BaseTable::Raw(rows) = &mut base_table
        {
            let mut counts = vec![0usize; num_bases];
            for sample_idx in 0..num_samples {
                let sample = encoded_data.get_sample(sample_idx).ok_or_else(|| {
                    EntroGdError::InvalidMetadata {
                        message: format!(
                            "failed to decode sample {} from encoded stream",
                            sample_idx
                        ),
                    }
                })?;

                let base_id = sample.id.load_le::<usize>();

                if base_id >= num_bases {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "encoded base id {} out of range for {} bases",
                            base_id, num_bases
                        ),
                    });
                }
                counts[base_id] += 1;
            }
            for (idx, count) in counts.into_iter().enumerate() {
                rows[idx].1 = count;
            }
        }

        let original_size_bits =
            n.checked_mul(chunk_size)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "original size overflow".to_string(),
                })?;
        let mut metadata = BitDataInfo::new(features, original_size_bits)?;
        metadata.set_condensed_sample_weights(if m == 0 { None } else { Some(weights.clone()) });

        Ok(CompressedData {
            encoded_data,
            condensed_sample_weights: if m == 0 { None } else { Some(weights) },
            base_table,
            layout: BaseLayoutInfo::from_bit_states(bit_states),
            entropy_sorted_column_order,
            metadata,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn save<P: AsRef<Path>>(&self, output_path: P) -> Result<PathBuf, EntroGdError> {
        let target = ensure_egd_extension(output_path.as_ref());
        fs::write(&target, &self.bytes)?;
        Ok(target)
    }
}

pub struct SaveEgdFile {
    pub output_path: PathBuf,
}

impl Filter for SaveEgdFile {
    type Input = CompressedData;
    type Output = PathBuf;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let egd_file = EgdFile::from_compressed_data(input)?;
        egd_file.save(&self.output_path)
    }
}

pub fn save_compressed_as_egd<P: AsRef<Path>>(
    compressed: CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    EgdFile::from_compressed_data(compressed)?.save(output_path)
}

pub struct LoadEgdFile {}

impl Filter for LoadEgdFile {
    type Input = PathBuf;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        EgdFile::load(input)?.to_compressed_data()
    }
}

pub struct DecodeDeltaBaseTable {}

impl Filter for DecodeDeltaBaseTable {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        if let BaseTable::Delta(delta) = input.base_table {
            input.base_table = BaseTable::Raw(delta.decode_rows()?);
        }
        Ok(input)
    }
}

pub fn load_compressed_from_egd<P: AsRef<Path>>(
    input_path: P,
) -> Result<CompressedData, EntroGdError> {
    EgdFile::load(input_path)?.to_compressed_data()
}

pub fn load_and_decompress_egd<P: AsRef<Path>>(input_path: P) -> Result<BitDataSet, EntroGdError> {
    let compressed = load_compressed_from_egd(input_path)?;
    decompress_file(compressed)
}
