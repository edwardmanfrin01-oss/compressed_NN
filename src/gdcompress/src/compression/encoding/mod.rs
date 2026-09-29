mod delta_base_table;
mod encoding_core;
mod huffman;
mod rle;

pub use self::delta_base_table::{
    DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, get_delta_codec, get_delta_codec_fixed,
};
pub use self::encoding_core::{
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodeData, EncodeDataOffsetRLE, EncodedData, HuffmanDeviationData,
    RleDeviationOffsetData,
};
pub use self::huffman::EncodeDataHuffman;
pub(crate) use self::rle::RLE_LONG_MAX;
