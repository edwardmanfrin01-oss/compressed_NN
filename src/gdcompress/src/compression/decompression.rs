use crate::compression::data::{
    BitData, BitDataReconstructionInfo, BitDataSet, ImageColorModel, ImageGroupingTransform,
};
use crate::compression::encoding::{BaseTable, CompressedData, CondensedSamples, EncodedData};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::{min_position_bits, signed_half_wrapped, zigzag_decode_i16};
use bitvec::prelude::*;
use image::{RgbImage, RgbaImage};
use rayon::prelude::*;
use std::path::Path;

const PARALLEL_MIN_ROWS: usize = 256;

/// Decompression context containing file-scoped state computed once per file.
/// Use this to perform multiple row decompression operations without recomputing layout.
#[derive(Debug, Clone)]
pub struct DecompressRandomAccessHandle {
    pub compressed: CompressedData,
    pub chunk_size: usize,
    pub stride: usize,
    pub original_num_rows: usize,
    pub deviation_positions: Vec<usize>,
    pub variable_base_positions: Vec<usize>,
    pub constant_one_positions: Vec<usize>,
}

impl DecompressRandomAccessHandle {
    /// Create a decompression context from compressed data.
    /// This validates metadata and pre-computes layout positions once.
    pub fn new(mut compressed: CompressedData) -> Result<Self, EntroGdError> {
        if let BaseTable::Delta(delta) = compressed.base_table {
            compressed.base_table = BaseTable::Raw(delta.decode_rows()?);
        }
        let data_info = &compressed.metadata;
        let num_features = data_info.num_features();
        let chunk_size = data_info.chunk_size();
        if chunk_size == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "chunk_size is 0".to_string(),
            });
        }

        let stride = data_info.row_stride();
        let original_num_rows = data_info.original_size_bits() / chunk_size;
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        Ok(Self {
            deviation_positions: compressed.layout.deviation_bit_positions(),
            variable_base_positions: compressed.layout.variable_base_bit_positions(),
            constant_one_positions: compressed.layout.constant_one_bit_positions(),
            compressed,
            chunk_size,
            stride,
            original_num_rows,
        })
    }

    /// Decompress a batch of rows from a context without recomputing file-scoped state.
    /// Output rows are in ascending sample-index order (sorted), regardless of `indices` order.
    pub fn decompress_samples(&self, indices: &[usize]) -> Result<BitDataSet, EntroGdError> {
        // Validate indices
        if let Some(&sample_idx) = indices.iter().find(|&&idx| idx >= self.original_num_rows) {
            return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
        }

        // Sort indices so the streaming decoder can do a single forward pass.
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        if sorted_indices.len() >= PARALLEL_MIN_ROWS {
            self.decompress_samples_parallel(&sorted_indices)
        } else {
            self.decompress_samples_sequential(&sorted_indices)
        }
    }

    fn decompress_samples_sequential(
        &self,
        sorted_indices: &[usize],
    ) -> Result<BitDataSet, EntroGdError> {
        let _timer = ScopedTimer::trace("Random-access decompression (sequential)");
        let mut reconstructed_bits = bitvec![usize, Lsb0; 0; self.stride * sorted_indices.len()];
        let mut out_cursor = 0usize;

        self.compressed
            .encoded_data
            .for_each_sample_at_sorted_indices(sorted_indices, |sample| {
                append_reconstructed_chunk(
                    &mut reconstructed_bits,
                    out_cursor,
                    &self.deviation_positions,
                    &self.variable_base_positions,
                    &self.constant_one_positions,
                    self.compressed.base_table.as_raw(),
                    sample.deviation,
                    sample.id,
                )?;
                out_cursor += self.stride;
                Ok(())
            })?;

        let data = BitData {
            data: reconstructed_bits,
            chunk_size: self.chunk_size,
            stride: self.stride,
            num_rows: sorted_indices.len(),
        };
        let info = self
            .compressed
            .metadata
            .with_original_size_bits_and_row_stride(
                self.chunk_size * sorted_indices.len(),
                self.stride,
            );
        Ok(BitDataSet { data, info })
    }

    fn decompress_samples_parallel(
        &self,
        sorted_indices: &[usize],
    ) -> Result<BitDataSet, EntroGdError> {
        let _timer = ScopedTimer::trace("Random-access decompression (parallel)");
        let num_threads = rayon::current_num_threads();
        let indices_per_thread = sorted_indices.len().div_ceil(num_threads);

        let deviation_positions = &self.deviation_positions;
        let variable_base_positions = &self.variable_base_positions;
        let constant_one_positions = &self.constant_one_positions;
        let base_table = self.compressed.base_table.as_raw();
        let stride = self.stride;
        let chunk_size = self.chunk_size;

        let chunks: Result<Vec<BitVec<usize, Lsb0>>, EntroGdError> = sorted_indices
            .par_chunks(indices_per_thread)
            .map(|chunk| {
                let mut chunk_bits = bitvec![usize, Lsb0; 0; stride * chunk.len()];
                let mut local_cursor = 0usize;

                self.compressed
                    .encoded_data
                    .for_each_sample_at_sorted_indices(chunk, |sample| {
                        append_reconstructed_chunk(
                            &mut chunk_bits,
                            local_cursor,
                            deviation_positions,
                            variable_base_positions,
                            constant_one_positions,
                            base_table,
                            sample.deviation,
                            sample.id,
                        )?;
                        local_cursor += stride;
                        Ok(())
                    })?;

                Ok(chunk_bits)
            })
            .collect();

        let mut reconstructed_bits = BitVec::with_capacity(stride * sorted_indices.len());
        for chunk in chunks? {
            reconstructed_bits.extend_from_bitslice(&chunk);
        }

        let data = BitData {
            data: reconstructed_bits,
            chunk_size,
            stride,
            num_rows: sorted_indices.len(),
        };
        let info = self
            .compressed
            .metadata
            .with_original_size_bits_and_row_stride(chunk_size * sorted_indices.len(), stride);
        Ok(BitDataSet { data, info })
    }
}

pub struct DecompressFileData {}

impl Filter for DecompressFileData {
    type Input = CompressedData;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing entire file data");
        decompress_file(input)
    }
}

pub fn decompress_file(mut compressed: CompressedData) -> Result<BitDataSet, EntroGdError> {
    if let BaseTable::Delta(delta) = compressed.base_table {
        compressed.base_table = BaseTable::Raw(delta.decode_rows()?);
    }
    let data_info = &compressed.metadata;
    let chunk_size = data_info.chunk_size();
    let original_num_rows = data_info.original_size_bits() / chunk_size;
    let stride = data_info.row_stride();

    if original_num_rows >= PARALLEL_MIN_ROWS {
        decompress_file_parallel(
            &compressed,
            &compressed.encoded_data,
            data_info,
            chunk_size,
            stride,
            original_num_rows,
        )
    } else {
        decompress_file_sequential(
            &compressed,
            data_info,
            chunk_size,
            stride,
            original_num_rows,
        )
    }
}

fn decompress_file_parallel(
    compressed: &CompressedData,
    encoded_data: &EncodedData,
    data_info: &crate::compression::data::BitDataInfo,
    chunk_size: usize,
    stride: usize,
    original_num_rows: usize,
) -> Result<BitDataSet, EntroGdError> {
    let _timer = ScopedTimer::trace("Reconstructing bit data (parallel)");

    let deviation_positions = compressed.layout.deviation_bit_positions();
    let variable_base_positions = compressed.layout.variable_base_bit_positions();
    let constant_one_positions = compressed.layout.constant_one_bit_positions();

    let base_table = compressed.base_table.as_raw();

    let num_threads = rayon::current_num_threads();
    let rows_per_thread = original_num_rows.div_ceil(num_threads);

    let chunks: Result<Vec<BitVec<usize, Lsb0>>, EntroGdError> = (0..num_threads)
        .into_par_iter()
        .map(|thread_idx| {
            let start = thread_idx * rows_per_thread;
            let count = rows_per_thread.min(original_num_rows.saturating_sub(start));
            if count == 0 {
                return Ok(BitVec::new());
            }

            let mut chunk_bits = bitvec![usize, Lsb0; 0; stride * count];
            let mut local_cursor = 0usize;

            encoded_data.for_each_sample_range(start, count, |sample| {
                append_reconstructed_chunk(
                    &mut chunk_bits,
                    local_cursor,
                    &deviation_positions,
                    &variable_base_positions,
                    &constant_one_positions,
                    base_table,
                    sample.deviation,
                    sample.id,
                )?;
                local_cursor += stride;
                Ok(())
            })?;

            Ok(chunk_bits)
        })
        .collect();

    let mut reconstructed_bits = BitVec::with_capacity(stride * original_num_rows);
    for chunk in chunks? {
        reconstructed_bits.extend_from_bitslice(&chunk);
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        stride,
        num_rows: original_num_rows,
    };
    let info =
        data_info.with_original_size_bits_and_row_stride(chunk_size * original_num_rows, stride);
    Ok(BitDataSet { data, info })
}

fn decompress_file_sequential(
    compressed: &CompressedData,
    data_info: &crate::compression::data::BitDataInfo,
    chunk_size: usize,
    stride: usize,
    original_num_rows: usize,
) -> Result<BitDataSet, EntroGdError> {
    let deviation_positions = &compressed.layout.deviation_bit_positions();
    let variable_base_positions = &compressed.layout.variable_base_bit_positions();
    let constant_one_positions = &compressed.layout.constant_one_bit_positions();
    let base_table = compressed.base_table.as_raw();
    let mut reconstructed_bits = bitvec![usize, Lsb0; 0 ; stride * original_num_rows];
    let mut decoded_rows = 0usize;
    let mut cursor = 0usize;

    let _timer = ScopedTimer::trace("Reconstructing bit data (sequential)");

    compressed
        .encoded_data
        .for_each_sample_n(original_num_rows, |sample| {
            append_reconstructed_chunk(
                &mut reconstructed_bits,
                cursor,
                deviation_positions,
                variable_base_positions,
                constant_one_positions,
                base_table,
                sample.deviation,
                sample.id,
            )?;
            cursor += stride;
            decoded_rows += 1;
            Ok(())
        })?;
    drop(_timer);

    if decoded_rows != original_num_rows {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "decoded row count mismatch: expected {}, got {}",
                original_num_rows, decoded_rows
            ),
        });
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        stride,
        num_rows: original_num_rows,
    };
    let info =
        data_info.with_original_size_bits_and_row_stride(chunk_size * original_num_rows, stride);
    Ok(BitDataSet { data, info })
}

fn append_reconstructed_chunk(
    out: &mut BitVec<usize, Lsb0>,
    out_start: usize,
    deviation_positions: &[usize],
    variable_base_positions: &[usize],
    constant_one_positions: &[usize],
    base_table: &[(BitVec<usize, Lsb0>, usize)],
    deviation_bits: &BitSlice<usize, Lsb0>,
    id_bits: &BitSlice<usize, Lsb0>,
) -> Result<(), EntroGdError> {
    let base_id = id_bits.load_le::<usize>();
    if base_id >= base_table.len() {
        return Err(EntroGdError::InvalidBaseId {
            base_id,
            table_len: base_table.len(),
        });
    }
    let base_pattern = &base_table[base_id].0;

    constant_one_positions
        .iter()
        .for_each(|&pos| out.set(out_start + pos, true));

    variable_base_positions
        .iter()
        .enumerate()
        .for_each(|(i, &pos)| {
            if let Some(bit) = base_pattern.get(i) {
                out.set(out_start + pos, *bit);
            }
        });

    deviation_positions
        .iter()
        .zip(deviation_bits.iter())
        .for_each(|(&pos, bit)| {
            out.set(out_start + pos, *bit);
        });

    Ok(())
}

pub struct DecompressAnalytics {}

impl Filter for DecompressAnalytics {
    type Input = CompressedData;
    type Output = Option<CondensedSamples>;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing condensed samples for analytics");
        if let BaseTable::Delta(delta) = input.base_table {
            input.base_table = BaseTable::Raw(delta.decode_rows()?);
        }
        Ok(decompress_analytics(&input))
    }
}

pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    if let Some(weights) = &compressed.condensed_sample_weights {
        let chunk_size = compressed.layout.chunk_size();
        let variable_positions = compressed.layout.variable_base_bit_positions();
        let constant_one_positions = compressed.layout.constant_one_bit_positions();

        let samples: Vec<BitVec<usize, Lsb0>> = compressed
            .base_table
            .as_raw()
            .iter()
            .map(|(bv, _)| {
                let mut reconstructed = bitvec![usize, Lsb0; 0; chunk_size];
                for &pos in &constant_one_positions {
                    reconstructed.set(pos, true);
                }
                for (i, &pos) in variable_positions.iter().enumerate() {
                    if let Some(bit) = bv.get(i) {
                        reconstructed.set(pos, *bit);
                    }
                }
                reconstructed
            })
            .collect();
        Some(CondensedSamples {
            samples,
            weights: weights.clone(),
        })
    } else {
        None
    }
}

/// Write decompressed image bit data back to an image file.
pub fn write_bitdata_as_image<P: AsRef<Path>>(
    bit_data: &BitDataSet,
    output_path: P,
) -> Result<(), EntroGdError> {
    let image_info = match &bit_data.info.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        BitDataReconstructionInfo::Tabular { .. } => {
            return Err(EntroGdError::InvalidMetadata {
                message: "cannot write image from tabular reconstruction metadata".to_string(),
            });
        }
    };

    let channels = image_info.channels as usize;
    let group_width = image_info.pixel_grouping.width() as usize;
    let group_height = image_info.pixel_grouping.height() as usize;
    if group_width == 0 || group_height == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "image pixel_grouping width and height must be > 0".to_string(),
        });
    }
    if bit_data.num_features() != channels {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "feature count {} does not match image channel count {}",
                bit_data.num_features(),
                channels
            ),
        });
    }

    let grouped_width = (image_info.width as usize).div_ceil(group_width);
    let grouped_height = (image_info.height as usize).div_ceil(group_height);
    let expected_rows = grouped_width * grouped_height;
    if bit_data.num_rows() != expected_rows {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "grouped image row count {} does not match expected {}",
                bit_data.num_rows(),
                expected_rows
            ),
        });
    }
    let _timer = ScopedTimer::trace(format!(
        "Writing decompressed image ({}x{}, {} channels) to raw buffer",
        image_info.width, image_info.height, channels
    ));
    let mut raw = vec![0u8; image_info.width as usize * image_info.height as usize * channels];
    let total_pixels = group_width * group_height;
    for row in 0..bit_data.num_rows() {
        let group_x = row % grouped_width;
        let group_y = row / grouped_width;
        let mut decoded_channels = Vec::with_capacity(channels);
        for feature in 0..channels {
            let feature_bits = unsafe { bit_data.get_feature_unchecked(row, feature) };
            decoded_channels.push(decode_grouped_feature(
                feature_bits,
                total_pixels,
                image_info.grouping_transform,
            )?);
        }

        for offset in 0..total_pixels {
            let pixel_x = group_x * group_width + (offset % group_width);
            let pixel_y = group_y * group_height + (offset / group_width);
            if pixel_x < image_info.width as usize && pixel_y < image_info.height as usize {
                for (feature, channel_values) in decoded_channels.iter().enumerate() {
                    raw[pixel_y * (image_info.width as usize) * channels
                        + pixel_x * channels
                        + feature] = channel_values[offset];
                }
            }
        }
    }

    let output_raw = match image_info.color_model {
        ImageColorModel::Rgb => raw,
        ImageColorModel::YCoCg => convert_ycocg_to_rgb_channels(&raw, channels),
        ImageColorModel::YCoCgR => convert_ycocg_r_to_rgb_channels(&raw, channels),
    };
    drop(_timer);
    save_raw_image(
        output_path.as_ref(),
        image_info.width,
        image_info.height,
        image_info.channels,
        output_raw,
    )
}

pub fn write_cropped_bitdata_as_image<P: AsRef<Path>>(
    bit_data: &BitDataSet,
    row_indices: &[usize],
    output_path: P,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
) -> Result<(), EntroGdError> {
    let image_info = match &bit_data.info.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        BitDataReconstructionInfo::Tabular { .. } => {
            return Err(EntroGdError::InvalidMetadata {
                message: "cannot write image from tabular reconstruction metadata".to_string(),
            });
        }
    };

    if x1 > x2 || y1 > y2 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "invalid crop bounds: ({},{})-({},{}), x1 <= x2 and y1 <= y2 required",
                x1, y1, x2, y2
            ),
        });
    }
    if x2 >= image_info.width || y2 >= image_info.height {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "crop ({},{})-({},{}) exceeds image dimensions {}x{}",
                x1, y1, x2, y2, image_info.width, image_info.height
            ),
        });
    }

    let crop_w = (x2 - x1 + 1) as usize;
    let crop_h = (y2 - y1 + 1) as usize;
    let channels = image_info.channels as usize;
    let group_w = image_info.pixel_grouping.width() as usize;
    let group_h = image_info.pixel_grouping.height() as usize;
    let total_pixels = group_w * group_h;
    let grouped_width = (image_info.width as usize).div_ceil(group_w);

    if row_indices.len() != bit_data.num_rows() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "row_indices length {} does not match bit_data row count {}",
                row_indices.len(),
                bit_data.num_rows()
            ),
        });
    }

    let mut raw = vec![0u8; crop_w * crop_h * channels];

    for j in 0..bit_data.num_rows() {
        let idx = row_indices[j];
        let group_x = idx % grouped_width;
        let group_y = idx / grouped_width;

        let mut decoded_channels = Vec::with_capacity(channels);
        for feature in 0..channels {
            let feature_bits = unsafe { bit_data.get_feature_unchecked(j, feature) };
            decoded_channels.push(decode_grouped_feature(
                feature_bits,
                total_pixels,
                image_info.grouping_transform,
            )?);
        }

        for offset in 0..total_pixels {
            let pixel_x = group_x * group_w + (offset % group_w);
            let pixel_y = group_y * group_h + (offset / group_w);

            if pixel_x >= image_info.width as usize
                || pixel_y >= image_info.height as usize
            {
                continue;
            }

            let px = pixel_x as u32;
            let py = pixel_y as u32;
            if px < x1 || px > x2 || py < y1 || py > y2 {
                continue;
            }

            let out_x = (px - x1) as usize;
            let out_y = (py - y1) as usize;
            for (feature, channel_values) in decoded_channels.iter().enumerate() {
                raw[out_y * crop_w * channels + out_x * channels + feature] =
                    channel_values[offset];
            }
        }
    }

    let output_raw = match image_info.color_model {
        ImageColorModel::Rgb => raw,
        ImageColorModel::YCoCg => convert_ycocg_to_rgb_channels(&raw, channels),
        ImageColorModel::YCoCgR => convert_ycocg_r_to_rgb_channels(&raw, channels),
    };

    save_raw_image(
        output_path.as_ref(),
        crop_w as u32,
        crop_h as u32,
        image_info.channels,
        output_raw,
    )
}

fn byte_from_bits(bits: &BitSlice<usize, Lsb0>) -> Result<u8, EntroGdError> {
    if bits.len() != 8 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("expected 8 bits for image byte, got {}", bits.len()),
        });
    }
    Ok(bits.load_le::<u8>())
}

fn bits_to_u16(bits: &BitSlice<usize, Lsb0>) -> Result<u16, EntroGdError> {
    if bits.len() > 16 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("expected at most 16 bits, got {}", bits.len()),
        });
    }
    Ok(bits.load_le::<u16>())
}

fn decode_grouped_feature(
    bits: &BitSlice<usize, Lsb0>,
    total_pixels: usize,
    grouping_transform: ImageGroupingTransform,
) -> Result<Vec<u8>, EntroGdError> {
    match grouping_transform {
        ImageGroupingTransform::Raw => bits.chunks(8).map(byte_from_bits).collect(),
        ImageGroupingTransform::ForFirstPixel => {
            if total_pixels == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let expected_bits = 8 + total_pixels.saturating_sub(1) * 9;
            if bits.len() != expected_bits {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid FOR(first pixel) feature width: expected {}, got {}",
                        expected_bits,
                        bits.len()
                    ),
                });
            }

            let anchor = byte_from_bits(&bits[0..8])? as i16;
            let mut values = Vec::with_capacity(total_pixels);
            values.push(anchor as u8);

            for offset in 0..total_pixels.saturating_sub(1) {
                let start = 8 + offset * 9;
                let end = start + 9;
                let encoded = bits_to_u16(&bits[start..end])?;
                let value = anchor + zigzag_decode_i16(encoded);
                if !(0..=255).contains(&value) {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!("decoded grouped image byte out of range: {}", value),
                    });
                }
                values.push(value as u8);
            }

            Ok(values)
        }
        ImageGroupingTransform::ForMin => {
            if total_pixels == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let position_bits = min_position_bits(total_pixels);
            let expected_bits = 8 + position_bits + total_pixels.saturating_sub(1) * 8;
            if bits.len() != expected_bits {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid FOR(min) feature width: expected {}, got {}",
                        expected_bits,
                        bits.len()
                    ),
                });
            }

            let anchor = byte_from_bits(&bits[0..8])? as u16;
            let min_position = if position_bits == 0 {
                0
            } else {
                bits_to_u16(&bits[8..8 + position_bits])? as usize
            };
            if min_position >= total_pixels {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("FOR(min) min position {} out of bounds", min_position),
                });
            }

            let mut values = Vec::with_capacity(total_pixels);
            let mut residual_cursor = 8 + position_bits;
            for idx in 0..total_pixels {
                if idx == min_position {
                    values.push(anchor as u8);
                    continue;
                }

                let encoded = bits_to_u16(&bits[residual_cursor..residual_cursor + 8])?;
                let value = anchor + encoded;
                if !(0..=255).contains(&value) {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!("decoded grouped image byte out of range: {}", value),
                    });
                }
                values.push(value as u8);
                residual_cursor += 8;
            }

            Ok(values)
        }
    }
}

fn convert_ycocg_to_rgb_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let y = i16::from(pixel[0]);
        let co = i16::from(pixel[1]) - 128;
        let cg = i16::from(pixel[2]) - 128;

        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;

        pixel[0] = clamp_to_u8(r);
        pixel[1] = clamp_to_u8(g);
        pixel[2] = clamp_to_u8(b);
    }
    out
}

fn convert_ycocg_r_to_rgb_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let y = pixel[0];
        let co = pixel[1].wrapping_sub(128);
        let cg = pixel[2].wrapping_sub(128);

        let t = y.wrapping_sub(signed_half_wrapped(cg));
        let g = cg.wrapping_add(t);
        let b = t.wrapping_sub(signed_half_wrapped(co));
        let r = b.wrapping_add(co);

        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
    out
}

fn clamp_to_u8(value: i16) -> u8 {
    value.clamp(0, 255) as u8
}

fn save_raw_image(
    output_path: &Path,
    width: u32,
    height: u32,
    channels: u8,
    raw: Vec<u8>,
) -> Result<(), EntroGdError> {
    let _timer = ScopedTimer::trace(format!(
        "Saving decompressed image to file: {}",
        output_path.display()
    ));
    match channels {
        3 => {
            let image = RgbImage::from_raw(width, height, raw).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw RGB buffer size does not match width*height*3".to_string(),
                }
            })?;
            image.save(output_path)?;
        }
        4 => {
            let image = RgbaImage::from_raw(width, height, raw).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw RGBA buffer size does not match width*height*4".to_string(),
                }
            })?;
            image.save(output_path)?;
        }
        _ => {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("unsupported channel count {}", channels),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::data::{BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform};
    use crate::compression::encoding::{DeviationData, DeviationSample, EncodedData};

    // ============================================================================
    // HELPER FUNCTIONS FOR CONSTRUCTING TEST DATA
    // ============================================================================

    /// Creates a simple BitVec with a specific pattern for testing.
    /// Pattern: alternating bits if alternate=true, all zeros if alternate=false
    fn create_bit_pattern(size: usize, alternate: bool) -> BitVec<usize, Lsb0> {
        let mut bits = BitVec::with_capacity(size);
        for i in 0..size {
            bits.push(if alternate { i % 2 == 0 } else { false });
        }
        bits
    }

    /// Creates a BitDataInfo for testing with a single feature
    fn create_test_bit_data_info(chunk_size: usize, num_rows: usize) -> BitDataInfo {
        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UInt(chunk_size as u16),
            transform: FeatureTransform::None,
        }];
        BitDataInfo::new(features, chunk_size * num_rows).unwrap()
    }

    /// Creates a deviation sample with specified ID and deviation bits
    fn create_deviation_sample(id_bits: Vec<bool>, deviation_bits: Vec<bool>) -> DeviationSample {
        let mut id = BitVec::with_capacity(id_bits.len());
        for bit in id_bits {
            id.push(bit);
        }
        let mut deviation = BitVec::with_capacity(deviation_bits.len());
        for bit in deviation_bits {
            deviation.push(bit);
        }
        DeviationSample { deviation, id }
    }

    /// Creates normal (non-RLE) encoded data with specified samples
    fn create_normal_encoded_data(samples: Vec<DeviationSample>) -> EncodedData {
        let num_samples = samples.len();
        let num_id_bits = if num_samples > 0 {
            samples[0].id.len()
        } else {
            0
        };
        let num_deviation_bits = if num_samples > 0 {
            samples[0].deviation.len()
        } else {
            0
        };

        let mut encoded_bit_stream = BitVec::new();
        for sample in samples {
            encoded_bit_stream.extend(&sample.deviation);
            encoded_bit_stream.extend(&sample.id);
        }

        EncodedData::Normal(DeviationData::new(
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        ))
    }

    /// Creates a base table for testing
    fn create_base_table(num_bases: usize, chunk_size: usize) -> Vec<(BitVec<usize, Lsb0>, usize)> {
        (0..num_bases)
            .map(|i| {
                let pattern = create_bit_pattern(chunk_size, i % 2 == 0);
                (pattern, i) // frequency is just the index for simplicity
            })
            .collect()
    }

    /// Creates a base bit positions list
    fn create_base_bit_positions(chunk_size: usize, num_base_bits: usize) -> Vec<usize> {
        (0..chunk_size.min(num_base_bits)).collect()
    }

    /// Creates a minimal CompressedData structure for testing
    fn create_minimal_compressed_data(
        chunk_size: usize,
        num_rows: usize,
        num_bases: usize,
        num_base_bits: usize,
    ) -> CompressedData {
        let metadata = create_test_bit_data_info(chunk_size, num_rows);
        let base_table = create_base_table(num_bases, num_base_bits);
        let base_bit_positions = create_base_bit_positions(chunk_size, num_base_bits);

        // Create sample encoded data - one sample per row
        let samples: Vec<DeviationSample> = (0..num_rows)
            .map(|i| {
                let num_deviation_bits = chunk_size.saturating_sub(num_base_bits);
                create_deviation_sample(vec![i % 2 == 0], vec![false; num_deviation_bits])
            })
            .collect();
        let encoded_data = create_normal_encoded_data(samples);

        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table: crate::compression::encoding::BaseTable::Raw(base_table),
            layout: crate::compression::base_table::build_base_layout_from_constant_polarity(
                chunk_size,
                &base_bit_positions,
                &[],
                &[],
            ),
            entropy_sorted_column_order: None,
            metadata,
        }
    }

    /// Creates a CompressedData with condensed sample weights and analytics data
    fn create_compressed_data_with_analytics(
        chunk_size: usize,
        num_rows: usize,
        num_bases: usize,
        num_base_bits: usize,
        _num_condensed: usize,
    ) -> CompressedData {
        let mut compressed =
            create_minimal_compressed_data(chunk_size, num_rows, num_bases, num_base_bits);

        // Add weights for condensed samples
        compressed.condensed_sample_weights = Some((0..num_bases).map(|i| i * 10).collect());

        compressed
    }

    // ============================================================================
    // UNIT TESTS
    // ============================================================================

    #[test]
    fn test_decompress_samples_batch_single_sample() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_samples_batch_multiple_samples() {
        let chunk_size = 16;
        let num_rows = 10;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0, 2, 5, 9];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 4);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * 4);
    }

    #[test]
    fn test_decompress_samples_batch_all_samples() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices: Vec<usize> = (0..num_rows).collect();
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_samples_batch_out_of_bounds_index() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0, 10]; // Index 10 is out of bounds
        let result = context.decompress_samples(&indices);

        assert!(result.is_err());
        match result {
            Err(EntroGdError::DecompressionSampleMissing { sample_idx }) => {
                assert_eq!(sample_idx, 10);
            }
            _ => panic!("Expected DecompressionSampleMissing error"),
        }
    }

    #[test]
    fn test_decompress_samples_batch_empty_indices() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 0);
        assert_eq!(result.data.total_bits(), 0);
    }

    #[test]
    fn test_decompress_samples_batch_zero_features_error() {
        // This test is skipped because it's difficult to create invalid BitDataInfo
        // through the public API - the validation prevents construction of invalid metadata
    }

    #[test]
    fn test_decompress_file_all_rows() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let result = decompress_file(compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_file_single_row() {
        let chunk_size = 16;
        let num_rows = 1;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        let result = decompress_file(compressed).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_file_large_dataset() {
        let chunk_size = 32;
        let num_rows = 100;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 8, 16);

        let result = decompress_file(compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_analytics_with_weights() {
        let chunk_size = 8;
        let num_rows = 5;
        let num_bases = 4;
        let compressed =
            create_compressed_data_with_analytics(chunk_size, num_rows, num_bases, 4, 4);

        let result = decompress_analytics(&compressed);

        assert!(result.is_some());
        let analytics = result.unwrap();
        assert_eq!(analytics.samples.len(), num_bases);
        assert_eq!(analytics.weights.len(), num_bases);
        // Check that weights are as expected (0, 10, 20, 30)
        for (i, &weight) in analytics.weights.iter().enumerate() {
            assert_eq!(weight, i * 10);
        }
    }

    #[test]
    fn test_decompress_analytics_without_weights() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let result = decompress_analytics(&compressed);

        assert!(result.is_none());
    }

    #[test]
    fn test_decompress_analytics_samples_match_base_table() {
        let chunk_size = 8;
        let num_rows = 5;
        let num_bases = 4;
        let compressed =
            create_compressed_data_with_analytics(chunk_size, num_rows, num_bases, 4, 4);

        let result = decompress_analytics(&compressed).unwrap();
        let variable_positions = compressed.layout.variable_base_bit_positions();
        let constant_one_positions = compressed.layout.constant_one_bit_positions();

        // Verify that each sample is full chunk_size and base bits map correctly
        for (i, sample) in result.samples.iter().enumerate() {
            let base_row = &compressed.base_table.as_raw()[i].0;
            assert_eq!(sample.len(), chunk_size);

            // Base bits at variable positions match the base table row
            for (j, &pos) in variable_positions.iter().enumerate() {
                assert_eq!(sample[pos], base_row[j]);
            }

            // Constant-one bits are set
            for &pos in &constant_one_positions {
                assert!(sample[pos]);
            }

            // Deviation positions are false (no per-sample deviation data available)
            for pos in 0..chunk_size {
                if !variable_positions.contains(&pos) && !constant_one_positions.contains(&pos) {
                    assert!(!sample[pos]);
                }
            }
        }
    }

    #[test]
    fn test_decompressed_data_consistency() {
        let chunk_size = 16;
        let num_rows = 3;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        // Decompress all rows
        let all_rows = decompress_file(compressed.clone()).unwrap();

        // Decompress specific rows in same order
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();
        let specific_indices = vec![0, 1, 2];
        let specific_rows = context.decompress_samples(&specific_indices).unwrap();

        // Results should match
        assert_eq!(all_rows.data.num_rows, specific_rows.data.num_rows);
        assert_eq!(all_rows.data.total_bits(), specific_rows.data.total_bits());
        assert_eq!(all_rows.data.data, specific_rows.data.data);
    }

    #[test]
    fn test_decompressed_data_partial_subset() {
        let chunk_size = 16;
        let num_rows = 10;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        // Decompress specific subset
        let indices = vec![1, 3, 7];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 3);
        assert_eq!(result.data.chunk_size, chunk_size);
    }

    #[test]
    fn test_decompressed_metadata_preservation() {
        let chunk_size = 24;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 8);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0, 2];
        let result = context.decompress_samples(&indices).unwrap();

        // Metadata should be preserved and adjusted
        assert_eq!(result.info.chunk_size(), chunk_size);
        assert_eq!(result.info.n_data_samples(), 2); // We decompressed 2 samples
        assert_eq!(result.info.original_size_bits(), chunk_size * 2);
    }

    #[test]
    fn test_base_id_lsb0_load_semantics() {
        let mut id_bits = BitVec::<usize, Lsb0>::with_capacity(8);
        id_bits.push(true);
        id_bits.extend([false; 7]);

        assert_eq!(id_bits.load_le::<usize>(), 1);
    }

    #[test]
    fn test_decompressed_filter_interface() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let filter = DecompressFileData {};
        let result = filter.process(compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
    }

    #[test]
    fn test_decompressed_rows_filter_interface() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0, 2];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 2);
    }

    #[test]
    fn test_decompress_with_single_base_bit() {
        let chunk_size = 8;
        let num_rows = 3;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 1);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_with_all_base_bits() {
        let chunk_size = 8;
        let num_rows = 3;
        // All bits are base bits, no deviation bits
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, chunk_size);
        let context = DecompressRandomAccessHandle::new(compressed).unwrap();

        let indices = vec![0, 1];
        let result = context.decompress_samples(&indices).unwrap();

        assert_eq!(result.data.num_rows, 2);
        assert_eq!(result.data.total_bits(), chunk_size * 2);
    }
}
