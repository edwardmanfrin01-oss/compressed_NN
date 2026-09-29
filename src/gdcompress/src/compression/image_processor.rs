use bitvec::prelude::*;
use image::DynamicImage;
use std::path::PathBuf;

pub use crate::compression::data::{ImageColorModel, ImageGroupingTransform};

use crate::ScopedTimer;
use crate::compression::data::{
    BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, DEFAULT_ALIGN_ROWS_TO_WORD,
    FeatureDataType, FeatureSpec, FeatureTransform, ImageReconstructionInfo, PixelGrouping,
    aligned_stride, append_row_padding,
};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::utils::{min_position_bits, signed_half_wrapped, zigzag_encode_i16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// 0 = sRGB + linear alpha
    SrgbWithLinearAlpha = 0,
    /// 1 = all linear
    Linear = 1,
}

impl ImageColorSpace {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuildImageBitDataSet {
    pub colorspace: ImageColorSpace,
    pub color_model: ImageColorModel,
    pub pixel_grouping: PixelGrouping,
    pub grouping_transform: ImageGroupingTransform,
    pub pad_rows_to_word: bool,
}

#[derive(Debug, Clone)]
struct ImageBuildInput {
    width: u32,
    height: u32,
    channels: u8,
    colorspace: u8,
    raw: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct ImageBuildOptions {
    color_model: ImageColorModel,
    pixel_grouping: PixelGrouping,
    grouping_transform: ImageGroupingTransform,
    pad_rows_to_word: bool,
}

impl Default for BuildImageBitDataSet {
    fn default() -> Self {
        Self {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::default(),
            grouping_transform: ImageGroupingTransform::Raw,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenImage;

impl Filter for OpenImage {
    type Input = PathBuf;
    type Output = DynamicImage;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::debug(format!("OpenImage for {}", input.display()));
        let decoded = image::open(&input)?;
        Ok(decoded)
    }
}

impl Filter for BuildImageBitDataSet {
    type Input = DynamicImage;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::debug("BuildImageBitDataSet".to_string());
        match input {
            DynamicImage::ImageRgb8(img) => build_image_bitdataset(
                ImageBuildInput {
                    width: img.width(),
                    height: img.height(),
                    channels: 3,
                    colorspace: self.colorspace.as_u8(),
                    raw: img.into_raw(),
                },
                ImageBuildOptions {
                    color_model: self.color_model,
                    pixel_grouping: self.pixel_grouping,
                    grouping_transform: self.grouping_transform,
                    pad_rows_to_word: self.pad_rows_to_word,
                },
            ),
            DynamicImage::ImageRgba8(img) => build_image_bitdataset(
                ImageBuildInput {
                    width: img.width(),
                    height: img.height(),
                    channels: 4,
                    colorspace: self.colorspace.as_u8(),
                    raw: img.into_raw(),
                },
                ImageBuildOptions {
                    color_model: self.color_model,
                    pixel_grouping: self.pixel_grouping,
                    grouping_transform: self.grouping_transform,
                    pad_rows_to_word: self.pad_rows_to_word,
                },
            ),
            other => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image color type for dataset ingestion: {:?} (expected RGB8 or RGBA8)",
                    other.color()
                ),
            }),
        }
    }
}

fn build_image_bitdataset(
    input: ImageBuildInput,
    options: ImageBuildOptions,
) -> Result<BitDataSet, EntroGdError> {
    let ImageBuildInput {
        width,
        height,
        channels,
        colorspace,
        mut raw,
    } = input;
    let ImageBuildOptions {
        color_model,
        pixel_grouping,
        grouping_transform,
        pad_rows_to_word,
    } = options;

    if !matches!(channels, 3 | 4) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported channel count {} (expected 3 or 4)", channels),
        });
    }
    if !matches!(colorspace, 0 | 1) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported colorspace {} (expected 0 or 1)", colorspace),
        });
    }
    if pixel_grouping.width() == 0 || pixel_grouping.height() == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "pixel_grouping width and height must be > 0".to_string(),
        });
    }

    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image dimensions overflow row count".to_string(),
        })?;
    let channels_usize = channels as usize;
    let expected_len =
        pixels
            .checked_mul(channels_usize)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image dimensions overflow raw data length".to_string(),
            })?;
    if raw.len() != expected_len {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "invalid raw image length: expected {}, got {}",
                expected_len,
                raw.len()
            ),
        });
    }

    let group_width = pixel_grouping.width() as usize;
    let group_height = pixel_grouping.height() as usize;
    let pixels_per_group =
        group_width
            .checked_mul(group_height)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "pixel grouping overflows total pixel count".to_string(),
            })?;
    let feature_bits = feature_bits_for_grouping(pixels_per_group, grouping_transform)?;
    let grouped_width = (width as usize).div_ceil(group_width);
    let grouped_height = (height as usize).div_ceil(group_height);
    let rows =
        grouped_width
            .checked_mul(grouped_height)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image dimensions overflow grouped row count".to_string(),
            })?;

    let features = vec![
        FeatureSpec {
            data_type: FeatureDataType::UInt(feature_bits as u16),
            transform: FeatureTransform::None,
        };
        channels_usize
    ];
    let reconstruction = BitDataReconstructionInfo::Image(ImageReconstructionInfo {
        width,
        height,
        channels,
        color_model,
        pixel_grouping,
        grouping_transform,
        colorspace,
    });

    let chunk_size =
        channels_usize
            .checked_mul(feature_bits)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image chunk size overflow".to_string(),
            })?;
    let stride = aligned_stride(chunk_size, pad_rows_to_word);
    let logical_total_bits =
        rows.checked_mul(chunk_size)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image logical bitdata size overflow".to_string(),
            })?;
    let storage_bits = rows
        .checked_mul(stride)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image padded bitdata size overflow".to_string(),
        })?;
    let info =
        BitDataInfo::new_with_reconstruction_info(features, logical_total_bits, reconstruction)?
            .with_original_size_bits_and_row_stride(logical_total_bits, stride);

    let _timer = ScopedTimer::trace("Encoding image into color model");
    match color_model {
        ImageColorModel::Rgb => {}
        ImageColorModel::YCoCg => convert_rgb_to_ycocg_channels(&mut raw, channels_usize),
        ImageColorModel::YCoCgR => convert_rgb_to_ycocg_r_channels(&mut raw, channels_usize),
    }
    drop(_timer);
    let width_usize = width as usize;
    let height_usize = height as usize;
    let _timer = ScopedTimer::trace("Encoding grouped/transformed pixel data into bitstream");
    let bitstream = match grouping_transform {
        ImageGroupingTransform::Raw => build_raw_transform_bitstream_from_raw(
            &raw,
            width_usize,
            height_usize,
            channels_usize,
            grouped_width,
            grouped_height,
            group_width,
            group_height,
            pixels_per_group,
            chunk_size,
            stride,
            storage_bits,
        ),
        ImageGroupingTransform::ForFirstPixel => {
            build_for_first_pixel_transform_bitstream_from_raw(
                &raw,
                width_usize,
                height_usize,
                channels_usize,
                grouped_width,
                grouped_height,
                group_width,
                group_height,
                pixels_per_group,
                chunk_size,
                stride,
                storage_bits,
            )
        }
        ImageGroupingTransform::ForMin => build_grouped_transform_bitstream_from_raw(
            &raw,
            width_usize,
            height_usize,
            channels_usize,
            grouped_width,
            grouped_height,
            group_width,
            group_height,
            pixels_per_group,
            grouping_transform,
            chunk_size,
            stride,
            storage_bits,
        ),
    };

    let data = BitData {
        data: bitstream,
        chunk_size,
        stride,
        num_rows: rows,
    };

    Ok(BitDataSet { data, info })
}

fn push_bits_u8(out: &mut BitVec<usize, Lsb0>, value: u8) {
    for shift in 0..8 {
        out.push(((value >> shift) & 1) == 1);
    }
}

fn push_bits_u16(out: &mut BitVec<usize, Lsb0>, value: u16, bit_count: usize) {
    for shift in 0..bit_count {
        out.push(((value >> shift) & 1) == 1);
    }
}

fn build_raw_transform_bitstream_from_raw(
    raw: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    grouped_width: usize,
    grouped_height: usize,
    group_width: usize,
    group_height: usize,
    pixels_per_group: usize,
    chunk_size: usize,
    stride: usize,
    storage_bits: usize,
) -> BitVec<usize, Lsb0> {
    let mut bitstream = BitVec::with_capacity(storage_bits);
    bitstream.resize(storage_bits, false);

    let _row_stride_values = width * channels;
    let row_padding_bits = stride.saturating_sub(chunk_size);

    let mut bit_cursor = 0usize;
    for group_y in 0..grouped_height {
        for group_x in 0..grouped_width {
            for channel in 0..channels {
                let mut grouped_values = vec![0u8; pixels_per_group];
                grouped_channel_values_into(
                    raw,
                    width,
                    height,
                    channels,
                    group_x,
                    group_y,
                    group_width,
                    group_height,
                    channel,
                    &mut grouped_values,
                );
                for value in grouped_values {
                    store_u8_le_at(&mut bitstream, bit_cursor, value);
                    bit_cursor += 8;
                }
            }
            bit_cursor += row_padding_bits;
        }
    }

    debug_assert_eq!(bit_cursor, storage_bits);
    bitstream
}

fn build_for_first_pixel_transform_bitstream_from_raw(
    raw: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    grouped_width: usize,
    grouped_height: usize,
    group_width: usize,
    group_height: usize,
    pixels_per_group: usize,
    chunk_size: usize,
    stride: usize,
    storage_bits: usize,
) -> BitVec<usize, Lsb0> {
    let mut bitstream = BitVec::with_capacity(storage_bits);
    bitstream.resize(storage_bits, false);

    let row_padding_bits = stride.saturating_sub(chunk_size);

    let mut bit_cursor = 0usize;
    for group_y in 0..grouped_height {
        for group_x in 0..grouped_width {
            for channel in 0..channels {
                let mut grouped_values = vec![0u8; pixels_per_group];
                grouped_channel_values_into(
                    raw,
                    width,
                    height,
                    channels,
                    group_x,
                    group_y,
                    group_width,
                    group_height,
                    channel,
                    &mut grouped_values,
                );
                let anchor = grouped_values.first().copied().unwrap_or(0);
                store_u8_le_at(&mut bitstream, bit_cursor, anchor);
                bit_cursor += 8;

                for value in grouped_values.iter().skip(1) {
                    let delta = *value as i16 - anchor as i16;
                    let encoded = zigzag_encode_i16(delta);
                    store_u16_le_at(&mut bitstream, bit_cursor, encoded, 9);
                    bit_cursor += 9;
                }
            }
            bit_cursor += row_padding_bits;
        }
    }

    debug_assert_eq!(bit_cursor, storage_bits);
    bitstream
}

fn build_grouped_transform_bitstream_from_raw(
    raw: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    grouped_width: usize,
    grouped_height: usize,
    group_width: usize,
    group_height: usize,
    pixels_per_group: usize,
    grouping_transform: ImageGroupingTransform,
    chunk_size: usize,
    stride: usize,
    storage_bits: usize,
) -> BitVec<usize, Lsb0> {
    let mut bitstream = BitVec::with_capacity(storage_bits);
    let row_padding_bits = stride.saturating_sub(chunk_size);

    for group_y in 0..grouped_height {
        for group_x in 0..grouped_width {
            for channel in 0..channels {
                let mut grouped_values = vec![0u8; pixels_per_group];
                grouped_channel_values_into(
                    raw,
                    width,
                    height,
                    channels,
                    group_x,
                    group_y,
                    group_width,
                    group_height,
                    channel,
                    &mut grouped_values,
                );
                encode_grouped_channel(&mut bitstream, &grouped_values, grouping_transform);
            }
            append_row_padding(&mut bitstream, row_padding_bits);
        }
    }

    bitstream
}

#[inline(always)]
fn store_u8_le_at(out: &mut BitVec<usize, Lsb0>, bit_cursor: usize, value: u8) {
    for shift in 0..8 {
        out.set(bit_cursor + shift, ((value >> shift) & 1) == 1);
    }
}

#[inline(always)]
fn store_u16_le_at(out: &mut BitVec<usize, Lsb0>, bit_cursor: usize, value: u16, bit_count: usize) {
    for shift in 0..bit_count {
        out.set(bit_cursor + shift, ((value >> shift) & 1) == 1);
    }
}

fn feature_bits_for_grouping(
    pixels_per_group: usize,
    grouping_transform: ImageGroupingTransform,
) -> Result<usize, EntroGdError> {
    match grouping_transform {
        ImageGroupingTransform::Raw => {
            pixels_per_group
                .checked_mul(8)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
        ImageGroupingTransform::ForFirstPixel => {
            if pixels_per_group == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let residual_bits = pixels_per_group
                .saturating_sub(1)
                .checked_mul(9)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows residual bit width".to_string(),
                })?;
            8usize
                .checked_add(residual_bits)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
        ImageGroupingTransform::ForMin => {
            if pixels_per_group == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let position_bits = min_position_bits(pixels_per_group);
            let residual_bits = pixels_per_group
                .saturating_sub(1)
                .checked_mul(8)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows residual bit width".to_string(),
                })?;
            8usize
                .checked_add(position_bits)
                .and_then(|bits| bits.checked_add(residual_bits))
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
    }
}

#[inline(always)]
fn grouped_channel_values_into(
    raw: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    group_x: usize,
    group_y: usize,
    group_width: usize,
    group_height: usize,
    channel: usize,
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), group_width * group_height);
    for (offset, value) in out.iter_mut().enumerate() {
        let offset_x = offset % group_width;
        let offset_y = offset / group_width;
        let pixel_x = group_x * group_width + offset_x;
        let pixel_y = group_y * group_height + offset_y;
        if pixel_x < width && pixel_y < height {
            let pixel_index = pixel_y * width + pixel_x;
            let raw_index = pixel_index * channels + channel;
            *value = raw[raw_index];
        } else {
            *value = 0;
        }
    }
}

fn encode_grouped_channel(
    out: &mut BitVec<usize, Lsb0>,
    values: &[u8],
    grouping_transform: ImageGroupingTransform,
) {
    match grouping_transform {
        ImageGroupingTransform::Raw => {
            for &value in values {
                push_bits_u8(out, value);
            }
        }
        ImageGroupingTransform::ForFirstPixel => {
            let anchor = values.first().copied().unwrap_or(0);
            encode_for_anchor(out, values, anchor);
        }
        ImageGroupingTransform::ForMin => {
            let (min_position, anchor) = values
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|&(idx, value)| (value, idx))
                .unwrap_or((0, 0));
            push_bits_u8(out, anchor);
            let position_bits = min_position_bits(values.len());
            push_bits_u16(out, min_position as u16, position_bits);
            for (idx, &value) in values.iter().enumerate() {
                if idx == min_position {
                    continue;
                }
                let delta = value as u16 - anchor as u16;
                push_bits_u16(out, delta, 8);
            }
        }
    }
}

fn encode_for_anchor(out: &mut BitVec<usize, Lsb0>, values: &[u8], anchor: u8) {
    push_bits_u8(out, anchor);
    for &value in values.iter().skip(1) {
        let delta = value as i16 - anchor as i16;
        let encoded = zigzag_encode_i16(delta);
        push_bits_u16(out, encoded, 9);
    }
}

fn convert_rgb_to_ycocg_channels(raw: &mut [u8], channels: usize) {
    tracing::warn!("By using the YCoCg color model, When converting back to RGB is lossy!");
    for pixel in raw.chunks_exact_mut(channels) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let y = (r >> 2).wrapping_add(g >> 1).wrapping_add(b >> 2);
        let co = ((i16::from(r >> 1) - i16::from(b >> 1)) + 128) as u8;
        let cg = ((-i16::from(r >> 2) + i16::from(g >> 1) - i16::from(b >> 2)) + 128) as u8;

        pixel[0] = y;
        pixel[1] = co;
        pixel[2] = cg;
    }
}

fn convert_rgb_to_ycocg_r_channels(raw: &mut [u8], channels: usize) {
    for pixel in raw.chunks_exact_mut(channels) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let co = r.wrapping_sub(b);
        let t = b.wrapping_add(signed_half_wrapped(co));
        let cg = g.wrapping_sub(t);
        let y = t.wrapping_add(signed_half_wrapped(cg));

        pixel[0] = y;
        pixel[1] = co.wrapping_add(128);
        pixel[2] = cg.wrapping_add(128);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::utils::zigzag_decode_i16;

    fn unique_tmp_path(name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.png", name, ts))
    }

    #[test]
    fn loads_rgb_image_as_rows_times_channels() {
        let path = unique_tmp_path("gdcompress_rgb");
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(bit_data.num_rows(), 2);
        assert_eq!(bit_data.num_features(), 3);
        assert_eq!(bit_data.chunk_size(), 24);
        assert_eq!(bit_data.info.original_size_bits(), 48);
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(1, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loads_rgba_image_with_four_features() {
        let path = unique_tmp_path("gdcompress_rgba");
        let mut img = RgbaImage::new(1, 2);
        img.put_pixel(0, 0, Rgba([1, 2, 3, 4]));
        img.put_pixel(0, 1, Rgba([5, 6, 7, 8]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(bit_data.num_rows(), 2);
        assert_eq!(bit_data.num_features(), 4);
        assert_eq!(bit_data.chunk_size(), 32);
        assert_eq!(bit_data.info.original_size_bits(), 64);
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 1,
                height: 2,
                channels: 4,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(1, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 1
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    fn bits_to_u16(bits: &BitSlice<usize, Lsb0>) -> u16 {
        bits.load_le::<u16>()
    }

    fn decode_grouped_feature(
        bits: &BitSlice<usize, Lsb0>,
        pixel_grouping: usize,
        grouping_transform: ImageGroupingTransform,
    ) -> Vec<u8> {
        match grouping_transform {
            ImageGroupingTransform::Raw => {
                bits.chunks(8).map(|chunk| chunk.load_le::<u8>()).collect()
            }
            ImageGroupingTransform::ForFirstPixel => {
                let anchor = bits_to_u16(&bits[0..8]) as i16;
                let mut values = Vec::with_capacity(pixel_grouping);
                values.push(anchor as u8);
                for offset in 0..pixel_grouping.saturating_sub(1) {
                    let start = 8 + offset * 9;
                    let end = start + 9;
                    let encoded = bits_to_u16(&bits[start..end]);
                    values.push((anchor + zigzag_decode_i16(encoded)) as u8);
                }
                values
            }
            ImageGroupingTransform::ForMin => {
                let anchor = bits_to_u16(&bits[0..8]);
                let position_bits = min_position_bits(pixel_grouping);
                let min_position = if position_bits == 0 {
                    0
                } else {
                    bits_to_u16(&bits[8..8 + position_bits]) as usize
                };
                let mut values = Vec::with_capacity(pixel_grouping);
                let mut residual_cursor = 8 + position_bits;
                for idx in 0..pixel_grouping {
                    if idx == min_position {
                        values.push(anchor as u8);
                    } else {
                        let encoded = bits_to_u16(&bits[residual_cursor..residual_cursor + 8]);
                        values.push((anchor + encoded) as u8);
                        residual_cursor += 8;
                    }
                }
                values
            }
        }
    }

    #[test]
    fn zigzag_encoding_matches_expected_mapping() {
        assert_eq!(zigzag_encode_i16(0), 0);
        assert_eq!(zigzag_encode_i16(-1), 1);
        assert_eq!(zigzag_encode_i16(1), 2);
        assert_eq!(zigzag_encode_i16(-2), 3);
        assert_eq!(zigzag_encode_i16(2), 4);
        assert_eq!(zigzag_encode_i16(-3), 5);
        assert_eq!(zigzag_encode_i16(3), 6);

        assert_eq!(zigzag_decode_i16(0), 0);
        assert_eq!(zigzag_decode_i16(1), -1);
        assert_eq!(zigzag_decode_i16(2), 1);
        assert_eq!(zigzag_decode_i16(3), -2);
        assert_eq!(zigzag_decode_i16(4), 2);
        assert_eq!(zigzag_decode_i16(5), -3);
        assert_eq!(zigzag_decode_i16(6), 3);
    }

    #[test]
    fn loads_grouped_rgb_pixels_into_wider_features() {
        let path = unique_tmp_path("gdcompress_grouped_rgb");
        let mut img = RgbImage::new(4, 1);
        img.put_pixel(0, 0, Rgb([1, 11, 21]));
        img.put_pixel(1, 0, Rgb([2, 12, 22]));
        img.put_pixel(2, 0, Rgb([3, 13, 23]));
        img.put_pixel(3, 0, Rgb([4, 14, 24]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping(4, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.num_features(), 3);
        assert_eq!(bit_data.feature_bits(0), 35);
        assert_eq!(bit_data.chunk_size(), 105);
        assert_eq!(bit_data.info.original_size_bits(), 105);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![11, 12, 13, 14]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![21, 22, 23, 24]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(4, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn zero_pads_partial_group_at_row_end() {
        let path = unique_tmp_path("gdcompress_grouped_rgb_partial");
        let mut img = RgbImage::new(3, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.put_pixel(2, 0, Rgb([70, 80, 90]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::new(4, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.feature_bits(0), 35);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![10, 40, 70, 0]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![20, 50, 80, 0]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![30, 60, 90, 0]
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loads_grouped_rgb_pixels_with_for_min() {
        let path = unique_tmp_path("gdcompress_grouped_rgb_for_min");
        let mut img = RgbImage::new(4, 1);
        img.put_pixel(0, 0, Rgb([9, 14, 21]));
        img.put_pixel(1, 0, Rgb([4, 18, 17]));
        img.put_pixel(2, 0, Rgb([7, 16, 19]));
        img.put_pixel(3, 0, Rgb([5, 15, 20]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::new(4, 1),
            grouping_transform: ImageGroupingTransform::ForMin,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.feature_bits(0), 34);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![9, 4, 7, 5]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![14, 18, 16, 15]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![21, 17, 19, 20]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(4, 1),
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn encodes_rgb_as_ycocg_channels() {
        let path = unique_tmp_path("gdcompress_ycocg_rgb");
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCg,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![19]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![118]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![129]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                color_model: ImageColorModel::YCoCg,
                ..
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn leaves_alpha_unchanged_for_rgba_ycocg() {
        let path = unique_tmp_path("gdcompress_ycocg_rgba");
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([100, 110, 120, 130]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::YCoCg,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 3),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![130]
        );

        let _ = std::fs::remove_file(path);
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

    #[test]
    fn encodes_rgb_as_ycocg_r_channels() {
        let path = unique_tmp_path("gdcompress_ycocg_r_rgb");
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![20]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![108]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![128]
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ycocg_r_roundtrip_is_reversible_for_sampled_rgb_space() {
        for r in (0u8..=255).step_by(17) {
            for g in (0u8..=255).step_by(17) {
                for b in (0u8..=255).step_by(17) {
                    let original = vec![r, g, b];
                    let mut transformed = original.clone();
                    convert_rgb_to_ycocg_r_channels(&mut transformed, 3);
                    let recovered = convert_ycocg_r_to_rgb_channels(&transformed, 3);
                    assert_eq!(recovered, original);
                }
            }
        }
    }

    #[test]
    fn leaves_alpha_unchanged_for_rgba_ycocg_r() {
        let path = unique_tmp_path("gdcompress_ycocg_r_rgba");
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([100, 110, 120, 130]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        };
        let bit_data = filter
            .process(OpenImage.process(path.clone()).unwrap())
            .unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 3),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![130]
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn honors_row_padding_flag_for_image_bitdata() {
        let path = unique_tmp_path("gdcompress_alignment_image");
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([1, 2, 3]));
        img.put_pixel(1, 0, Rgb([4, 5, 6]));
        img.save(&path).unwrap();

        let compact = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: false,
        }
        .process(OpenImage.process(path.clone()).unwrap())
        .unwrap();
        assert_eq!(compact.chunk_size(), 24);
        assert_eq!(compact.data.stride, 24);

        let padded = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: PixelGrouping::new(1, 1),
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: true,
        }
        .process(OpenImage.process(path.clone()).unwrap())
        .unwrap();
        assert_eq!(padded.chunk_size(), 24);
        assert_eq!(padded.data.stride, 64);

        let _ = std::fs::remove_file(path);
    }
}
