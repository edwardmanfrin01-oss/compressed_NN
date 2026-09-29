use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum EntroGdError {
    Io(std::io::Error),
    Polars(polars::prelude::PolarsError),
    DataLoad { message: String },
    Image(image::ImageError),
    BitSliceLengthMismatch { expected: usize, actual: usize },
    DecompressionSampleMissing { sample_idx: usize },
    InvalidBaseId { base_id: usize, table_len: usize },
    InvalidMetadata { message: String },
    InvalidFeatureSpec { message: String },
    InvalidDataType { message: String },
}

impl Display for EntroGdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EntroGdError::Io(err) => write!(f, "IO error: {}", err),
            EntroGdError::Polars(err) => write!(f, "Polars error: {}", err),
            EntroGdError::DataLoad { message } => {
                write!(f, "Data load error: {}", message)
            }
            EntroGdError::Image(err) => write!(f, "Image error: {}", err),
            EntroGdError::BitSliceLengthMismatch { expected, actual } => write!(
                f,
                "Bit slice length mismatch (expected {}, got {})",
                expected, actual
            ),
            EntroGdError::DecompressionSampleMissing { sample_idx } => {
                write!(f, "Failed to retrieve sample at index {}", sample_idx)
            }
            EntroGdError::InvalidBaseId { base_id, table_len } => write!(
                f,
                "Invalid base ID: {} (base table has {} entries)",
                base_id, table_len
            ),
            EntroGdError::InvalidMetadata { message } => {
                write!(f, "Invalid compression metadata: {}", message)
            }
            EntroGdError::InvalidFeatureSpec { message } => {
                write!(f, "Invalid feature specification: {}", message)
            }
            EntroGdError::InvalidDataType { message } => {
                write!(f, "Invalid data type: {}", message)
            }
        }
    }
}

impl Error for EntroGdError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EntroGdError::Io(err) => Some(err),
            EntroGdError::Polars(err) => Some(err),
            EntroGdError::Image(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EntroGdError {
    fn from(err: std::io::Error) -> Self {
        EntroGdError::Io(err)
    }
}

impl From<polars::prelude::PolarsError> for EntroGdError {
    fn from(err: polars::prelude::PolarsError) -> Self {
        EntroGdError::Polars(err)
    }
}

impl From<image::ImageError> for EntroGdError {
    fn from(err: image::ImageError) -> Self {
        EntroGdError::Image(err)
    }
}
