use std::fs;
use std::path::{Path, PathBuf};

use super::egd::EgdFile;
use super::path_utils::ensure_igd_extension;

use crate::ScopedTimer;
use crate::compression::data::{
    BitDataReconstructionInfo, BitDataSet, ImageColorModel, ImageGroupingTransform,
    ImageReconstructionInfo, PixelGrouping,
};
use crate::compression::decompression::{decompress_file, write_bitdata_as_image};
use crate::compression::encoding::CompressedData;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;

pub const IMAGE_MAGIC_BYTES: [u8; 3] = *b"IGD";
pub const IMAGE_FORMAT_VERSION: u8 = 1;

/// In-memory IGD file contents (image + compressed payload) that can be saved to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgdFile {
    bytes: Vec<u8>,
}

impl IgdFile {
    pub fn from_compressed_data(compressed: CompressedData) -> Result<Self, EntroGdError> {
        let image_info = match compressed.metadata.reconstruction {
            BitDataReconstructionInfo::Image(info) => info,
            _ => {
                return Err(EntroGdError::InvalidMetadata {
                    message: "cannot build IGD from non-image reconstruction metadata".to_string(),
                });
            }
        };

        let egd_payload = EgdFile::from_compressed_data(compressed)?;
        let payload = egd_payload.as_bytes();
        let payload_len =
            u64::try_from(payload.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "IGD payload length does not fit into u64".to_string(),
            })?;

        let mut bytes = Vec::with_capacity(32 + payload.len());
        bytes.extend_from_slice(&IMAGE_MAGIC_BYTES);
        bytes.push(IMAGE_FORMAT_VERSION);
        bytes.extend_from_slice(&image_info.width.to_le_bytes());
        bytes.extend_from_slice(&image_info.height.to_le_bytes());
        bytes.push(image_info.channels);
        bytes.push(image_info.colorspace);
        bytes.push(image_info.color_model.as_u8());
        bytes.push(image_info.grouping_transform.as_u8());
        bytes.extend_from_slice(&image_info.pixel_grouping.width().to_le_bytes());
        bytes.extend_from_slice(&image_info.pixel_grouping.height().to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(payload);

        Ok(IgdFile { bytes })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        IgdFile { bytes }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EntroGdError> {
        let bytes = fs::read(path)?;
        Ok(IgdFile { bytes })
    }

    pub fn to_compressed_data(&self) -> Result<CompressedData, EntroGdError> {
        const HEADER_LEN: usize = 32;
        if self.bytes.len() < HEADER_LEN {
            return Err(EntroGdError::InvalidMetadata {
                message: "IGD file too short".to_string(),
            });
        }

        if self.bytes[0..3] != IMAGE_MAGIC_BYTES {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid magic bytes (expected IGD)".to_string(),
            });
        }
        if self.bytes[3] != IMAGE_FORMAT_VERSION {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported IGD format version {} (expected {})",
                    self.bytes[3], IMAGE_FORMAT_VERSION
                ),
            });
        }

        let width =
            u32::from_le_bytes([self.bytes[4], self.bytes[5], self.bytes[6], self.bytes[7]]);
        let height =
            u32::from_le_bytes([self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11]]);
        let channels = self.bytes[12];
        let colorspace = self.bytes[13];
        let color_model = ImageColorModel::from_u8(self.bytes[14])?;
        if !matches!(channels, 3 | 4) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("unsupported channel count {} in IGD metadata", channels),
            });
        }

        let grouping_transform = ImageGroupingTransform::from_u8(self.bytes[15])?;

        let pixel_grouping_width = u32::from_le_bytes([
            self.bytes[16],
            self.bytes[17],
            self.bytes[18],
            self.bytes[19],
        ]);
        let pixel_grouping_height = u32::from_le_bytes([
            self.bytes[20],
            self.bytes[21],
            self.bytes[22],
            self.bytes[23],
        ]);
        if pixel_grouping_width == 0 || pixel_grouping_height == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "IGD pixel_grouping width and height must be > 0".to_string(),
            });
        }

        let payload_len = u64::from_le_bytes([
            self.bytes[24],
            self.bytes[25],
            self.bytes[26],
            self.bytes[27],
            self.bytes[28],
            self.bytes[29],
            self.bytes[30],
            self.bytes[31],
        ]) as usize;

        let payload_start = HEADER_LEN;
        let payload_end = payload_start.checked_add(payload_len).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "IGD payload length overflow".to_string(),
            }
        })?;
        if payload_end != self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "IGD payload length does not match file size".to_string(),
            });
        }

        let payload = self.bytes[payload_start..payload_end].to_vec();
        let mut compressed = EgdFile::from_bytes(payload).to_compressed_data()?;
        if compressed.metadata.num_features() != channels as usize {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "IGD channels {} do not match compressed feature count {}",
                    channels,
                    compressed.metadata.num_features()
                ),
            });
        }
        compressed.metadata.reconstruction =
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width,
                height,
                channels,
                color_model,
                pixel_grouping: PixelGrouping::new(pixel_grouping_width, pixel_grouping_height),
                grouping_transform,
                colorspace,
            });

        Ok(compressed)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn save<P: AsRef<Path>>(&self, output_path: P) -> Result<PathBuf, EntroGdError> {
        let target = ensure_igd_extension(output_path.as_ref());
        fs::write(&target, &self.bytes)?;
        Ok(target)
    }
}

pub struct SaveIgdFile {
    pub output_path: PathBuf,
}

impl Filter for SaveIgdFile {
    type Input = CompressedData;
    type Output = PathBuf;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Saving compressed data as IGD file");
        let igd_file = IgdFile::from_compressed_data(input)?;
        igd_file.save(&self.output_path)
    }
}

pub struct LoadIgdFile {}

impl Filter for LoadIgdFile {
    type Input = PathBuf;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Loading compressed data from IGD file");
        IgdFile::load(input)?.to_compressed_data()
    }
}

pub fn save_compressed_as_igd<P: AsRef<Path>>(
    compressed: CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    IgdFile::from_compressed_data(compressed)?.save(output_path)
}

pub fn load_compressed_from_igd<P: AsRef<Path>>(
    input_path: P,
) -> Result<CompressedData, EntroGdError> {
    IgdFile::load(input_path)?.to_compressed_data()
}

/// Load an `.igd` file and fully decompress its payload into bit data.
pub fn load_and_decompress_igd<P: AsRef<Path>>(input_path: P) -> Result<BitDataSet, EntroGdError> {
    let compressed = load_compressed_from_igd(input_path)?;
    decompress_file(compressed)
}

/// Load an `.igd` file, decompress it, and regenerate the image file.
pub fn decompress_igd_to_image<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
) -> Result<PathBuf, EntroGdError> {
    let bit_data = load_and_decompress_igd(input_path)?;
    let target = output_path.as_ref().to_path_buf();
    write_bitdata_as_image(&bit_data, &target)?;
    Ok(target)
}
