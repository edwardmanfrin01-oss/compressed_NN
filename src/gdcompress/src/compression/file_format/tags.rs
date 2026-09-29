use crate::compression::data::FeatureDataType;
use crate::error::EntroGdError;

pub(super) const ENCODING_TAG_NORMAL: u8 = 0;
pub(super) const ENCODING_TAG_RLE_OFFSET: u8 = 2;
pub(super) const ENCODING_TAG_HUFFMAN_BASE_ID_ONLY: u8 = 3;

pub(super) const BASE_TABLE_TAG_RAW: u8 = 0;
pub(crate) const BASE_TABLE_TAG_DELTA_UNARY: u8 = 1;
pub(crate) const BASE_TABLE_TAG_DELTA_FIXED: u8 = 2;

pub(super) fn encode_data_type(data_type: FeatureDataType) -> u8 {
    match data_type {
        FeatureDataType::Int8 => 0,
        FeatureDataType::Int16 => 1,
        FeatureDataType::Int32 => 2,
        FeatureDataType::Int64 => 3,
        FeatureDataType::UInt8 => 4,
        FeatureDataType::UInt16 => 5,
        FeatureDataType::UInt32 => 6,
        FeatureDataType::UInt64 => 7,
        FeatureDataType::Float16 => 8,
        FeatureDataType::Float32 => 9,
        FeatureDataType::Float64 => 10,
        FeatureDataType::UInt128 => 11,
        FeatureDataType::UInt(_) => 12,
    }
}

pub(super) fn decode_data_type(tag: u8) -> Result<FeatureDataType, EntroGdError> {
    match tag {
        0 => Ok(FeatureDataType::Int8),
        1 => Ok(FeatureDataType::Int16),
        2 => Ok(FeatureDataType::Int32),
        3 => Ok(FeatureDataType::Int64),
        4 => Ok(FeatureDataType::UInt8),
        5 => Ok(FeatureDataType::UInt16),
        6 => Ok(FeatureDataType::UInt32),
        7 => Ok(FeatureDataType::UInt64),
        8 => Ok(FeatureDataType::Float16),
        9 => Ok(FeatureDataType::Float32),
        10 => Ok(FeatureDataType::Float64),
        11 => Ok(FeatureDataType::UInt128),
        12 => Ok(FeatureDataType::UInt(0)),
        _ => Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported feature data type tag {}", tag),
        }),
    }
}
