use polars::prelude::{CsvWriter, DataType, SerWriter};
use std::fs;
use std::path::{Path, PathBuf};

use super::egd::EgdFile;
use super::path_utils::{ensure_csv_extension, ensure_tgd_extension};

use crate::ScopedTimer;
use crate::compression::data::{BitDataReconstructionInfo, BitDataSet};
use crate::compression::decompression::decompress_file;
use crate::compression::encoding::CompressedData;
use crate::compression::tabular_processor::reconstruct_to_dataframe;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;

pub const TGD_MAGIC_BYTES: [u8; 3] = *b"TGD";
pub const TGD_FORMAT_VERSION: u8 = 1;

fn encode_polars_dtype(dt: &DataType) -> u8 {
    match dt {
        DataType::Float16 => 0,
        DataType::Float32 => 1,
        DataType::Float64 => 2,
        DataType::Int8 => 3,
        DataType::Int16 => 4,
        DataType::Int32 => 5,
        DataType::Int64 => 6,
        DataType::UInt8 => 7,
        DataType::UInt16 => 8,
        DataType::UInt32 => 9,
        DataType::UInt64 => 10,
        _ => 255,
    }
}

fn decode_polars_dtype(tag: u8) -> DataType {
    match tag {
        0 => DataType::Float16,
        1 => DataType::Float32,
        2 => DataType::Float64,
        3 => DataType::Int8,
        4 => DataType::Int16,
        5 => DataType::Int32,
        6 => DataType::Int64,
        7 => DataType::UInt8,
        8 => DataType::UInt16,
        9 => DataType::UInt32,
        10 => DataType::UInt64,
        _ => DataType::Float64,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TgdFile {
    bytes: Vec<u8>,
}

impl TgdFile {
    pub fn from_compressed_data(compressed: CompressedData) -> Result<Self, EntroGdError> {
        let (column_names, original_dtypes) = match &compressed.metadata.reconstruction {
            BitDataReconstructionInfo::Tabular {
                column_names,
                original_dtypes,
            } => (column_names.clone(), original_dtypes.clone()),
            _ => {
                return Err(EntroGdError::InvalidMetadata {
                    message:
                        "cannot build TGD from non-tabular reconstruction metadata".to_string(),
                });
            }
        };

        if column_names.len() != compressed.metadata.num_features() {
            return Err(EntroGdError::InvalidMetadata {
                message: "column names count does not match feature count".to_string(),
            });
        }
        if original_dtypes.len() != compressed.metadata.num_features() {
            return Err(EntroGdError::InvalidMetadata {
                message: "dtype count does not match feature count".to_string(),
            });
        }

        let num_columns = column_names.len() as u32;
        let egd_payload = EgdFile::from_compressed_data(compressed)?;
        let payload = egd_payload.as_bytes();
        let payload_len =
            u64::try_from(payload.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "TGD payload length does not fit into u64".to_string(),
            })?;

        // Compute names section size
        let mut names_bytes = Vec::new();
        for name in &column_names {
            let name_bytes = name.as_bytes();
            let len =
                u16::try_from(name_bytes.len()).map_err(|_| EntroGdError::InvalidMetadata {
                    message: format!("column name '{}' too long for u16", name),
                })?;
            names_bytes.extend_from_slice(&len.to_le_bytes());
            names_bytes.extend_from_slice(name_bytes);
        }
        let names_len =
            u32::try_from(names_bytes.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "names section too large for u32".to_string(),
            })?;

        // Dtypes section: one u8 tag per column
        let dtypes_len = num_columns;
        let mut dtypes_bytes = Vec::with_capacity(dtypes_len as usize);
        for dt in &original_dtypes {
            dtypes_bytes.push(encode_polars_dtype(dt));
        }

        // 24 = magic(3) + version(1) + num_columns(4) + names_len(4) + dtypes_len(4) + payload_len(8)
        let total = 24 + names_bytes.len() + dtypes_bytes.len() + payload.len();
        let mut bytes = Vec::with_capacity(total);

        bytes.extend_from_slice(&TGD_MAGIC_BYTES);
        bytes.push(TGD_FORMAT_VERSION);
        bytes.extend_from_slice(&num_columns.to_le_bytes());
        bytes.extend_from_slice(&names_len.to_le_bytes());
        bytes.extend_from_slice(&names_bytes);
        bytes.extend_from_slice(&dtypes_len.to_le_bytes());
        bytes.extend_from_slice(&dtypes_bytes);
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(payload);

        Ok(TgdFile { bytes })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        TgdFile { bytes }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EntroGdError> {
        let bytes = fs::read(path)?;
        Ok(TgdFile { bytes })
    }

    pub fn to_compressed_data(&self) -> Result<CompressedData, EntroGdError> {
        if self.bytes.len() < 24 {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD file too short for outer header".to_string(),
            });
        }

        if self.bytes[0..3] != TGD_MAGIC_BYTES {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid magic bytes (expected TGD)".to_string(),
            });
        }
        if self.bytes[3] != TGD_FORMAT_VERSION {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported TGD format version {} (expected {})",
                    self.bytes[3], TGD_FORMAT_VERSION
                ),
            });
        }

        let num_columns = u32::from_le_bytes([
            self.bytes[4],
            self.bytes[5],
            self.bytes[6],
            self.bytes[7],
        ]) as usize;

        let names_len = u32::from_le_bytes([
            self.bytes[8],
            self.bytes[9],
            self.bytes[10],
            self.bytes[11],
        ]) as usize;

        let names_start: usize = 12;
        let names_end = names_start
            .checked_add(names_len)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "TGD names section overflow".to_string(),
            })?;
        if names_end > self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD file too short for names section".to_string(),
            });
        }

        let mut column_names = Vec::with_capacity(num_columns);
        let mut cursor = names_start;
        for _ in 0..num_columns {
            if cursor + 2 > names_end {
                return Err(EntroGdError::InvalidMetadata {
                    message: "unexpected end of TGD names section".to_string(),
                });
            }
            let name_len = u16::from_le_bytes([
                self.bytes[cursor],
                self.bytes[cursor + 1],
            ]) as usize;
            cursor += 2;
            if cursor + name_len > names_end {
                return Err(EntroGdError::InvalidMetadata {
                    message: "column name exceeds TGD names section".to_string(),
                });
            }
            let name_bytes = &self.bytes[cursor..cursor + name_len];
            let name =
                String::from_utf8(name_bytes.to_vec()).map_err(|_| EntroGdError::InvalidMetadata {
                    message: "invalid UTF-8 in TGD column name".to_string(),
                })?;
            column_names.push(name);
            cursor += name_len;
        }

        let dtypes_offset = names_end;
        let dtypes_len = if dtypes_offset + 4 > self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD file too short for dtypes_len field".to_string(),
            });
        } else {
            u32::from_le_bytes([
                self.bytes[dtypes_offset],
                self.bytes[dtypes_offset + 1],
                self.bytes[dtypes_offset + 2],
                self.bytes[dtypes_offset + 3],
            ]) as usize
        };

        if dtypes_len != num_columns {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "TGD dtypes count {} does not match num_columns {}",
                    dtypes_len, num_columns
                ),
            });
        }

        let dtypes_start = dtypes_offset + 4;
        let dtypes_end = dtypes_start
            .checked_add(dtypes_len)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "TGD dtypes section overflow".to_string(),
            })?;
        if dtypes_end > self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD file too short for dtypes section".to_string(),
            });
        }

        let mut original_dtypes = Vec::with_capacity(num_columns);
        for i in 0..num_columns {
            original_dtypes.push(decode_polars_dtype(self.bytes[dtypes_start + i]));
        }

        let payload_len_offset = dtypes_end;
        if payload_len_offset + 8 > self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD file too short for payload_len field".to_string(),
            });
        }

        let payload_len = u64::from_le_bytes([
            self.bytes[payload_len_offset],
            self.bytes[payload_len_offset + 1],
            self.bytes[payload_len_offset + 2],
            self.bytes[payload_len_offset + 3],
            self.bytes[payload_len_offset + 4],
            self.bytes[payload_len_offset + 5],
            self.bytes[payload_len_offset + 6],
            self.bytes[payload_len_offset + 7],
        ]) as usize;

        let payload_start = payload_len_offset + 8;
        let payload_end = payload_start
            .checked_add(payload_len)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "TGD payload length overflow".to_string(),
            })?;
        if payload_end != self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "TGD payload length does not match file size".to_string(),
            });
        }

        let payload = self.bytes[payload_start..payload_end].to_vec();
        let mut compressed = EgdFile::from_bytes(payload).to_compressed_data()?;

        if compressed.metadata.num_features() != num_columns {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "TGD num_columns {} does not match compressed feature count {}",
                    num_columns,
                    compressed.metadata.num_features()
                ),
            });
        }

        compressed.metadata.reconstruction = BitDataReconstructionInfo::Tabular {
            column_names,
            original_dtypes,
        };

        Ok(compressed)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn save<P: AsRef<Path>>(&self, output_path: P) -> Result<PathBuf, EntroGdError> {
        let target = ensure_tgd_extension(output_path.as_ref());
        fs::write(&target, &self.bytes)?;
        Ok(target)
    }
}

pub struct SaveTgdFile {
    pub output_path: PathBuf,
}

impl Filter for SaveTgdFile {
    type Input = CompressedData;
    type Output = PathBuf;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Saving compressed data as TGD file");
        let tgd_file = TgdFile::from_compressed_data(input)?;
        tgd_file.save(&self.output_path)
    }
}

pub struct LoadTgdFile {}

impl Filter for LoadTgdFile {
    type Input = PathBuf;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Loading compressed data from TGD file");
        TgdFile::load(input)?.to_compressed_data()
    }
}

pub fn save_compressed_as_tgd<P: AsRef<Path>>(
    compressed: CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    TgdFile::from_compressed_data(compressed)?.save(output_path)
}

pub fn load_compressed_from_tgd<P: AsRef<Path>>(
    input_path: P,
) -> Result<CompressedData, EntroGdError> {
    TgdFile::load(input_path)?.to_compressed_data()
}

pub fn load_and_decompress_tgd<P: AsRef<Path>>(input_path: P) -> Result<BitDataSet, EntroGdError> {
    let compressed = load_compressed_from_tgd(input_path)?;
    decompress_file(compressed)
}

pub fn decompress_tgd_to_csv<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
) -> Result<PathBuf, EntroGdError> {
    let bit_data = load_and_decompress_tgd(input_path)?;
    let mut df = reconstruct_to_dataframe(&bit_data)?;
    let target = ensure_csv_extension(output_path.as_ref());
    let mut f = std::fs::File::create(&target)?;
    CsvWriter::new(&mut f).finish(&mut df)?;
    Ok(target)
}
