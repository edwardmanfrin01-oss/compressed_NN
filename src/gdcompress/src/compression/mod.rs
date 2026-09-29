pub mod base_bits;
pub mod base_selection;
pub mod base_table;
pub mod condensed_samples;
pub mod data;
pub mod decompression;
#[path = "encoding/mod.rs"]
pub mod encoding;
pub mod entropy;
#[path = "file_format/mod.rs"]
pub mod file_format;
pub mod image_processor;
pub mod tabular_processor;

pub use base_bits::{BaseBit, BaseBitGroups, BaseBitHyperLogLogCount};
pub use base_selection::{
    BaseBitImpl, BaseSelectionContext, SelectBases, SelectBasesAdaptive, SelectBasesDebug,
    SelectBasesProfileAllBits, SelectBasesThreshold,
};
pub use base_table::{BuildBaseTable, BuildSortedBaseTable, PreEncodeContext};
pub use condensed_samples::GenCondensedSamples;
pub use data::{
    BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo, BitDataSet,
    DEFAULT_ALIGN_ROWS_TO_WORD, FeatureDataType, FeatureSpec, FeatureTransform, ImageColorModel,
    ImageGroupingTransform, ImageReconstructionInfo, PixelGrouping, reconstruct_feature_value,
};
pub use decompression::{
    DecompressAnalytics, DecompressFileData, DecompressRandomAccessHandle, decompress_analytics,
    decompress_file, write_bitdata_as_image, write_cropped_bitdata_as_image,
};
pub use encoding::{
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodedData, RleDeviationOffsetData,
};
pub use encoding::{
    DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, EncodeData, EncodeDataHuffman,
    EncodeDataOffsetRLE,
};
pub use entropy::{Entropy, EntropyBitScore, EntropyScoredContext, calculate_entropy};
pub use file_format::{
    DecodeDeltaBaseTable, EgdFile, FORMAT_VERSION, IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES,
    IgdFile, LoadEgdFile, LoadIgdFile, LoadTgdFile, MAGIC_BYTES, SaveEgdFile, SaveIgdFile,
    SaveTgdFile, TGD_FORMAT_VERSION, TGD_MAGIC_BYTES, TgdFile, decompress_igd_to_image,
    decompress_tgd_to_csv, load_and_decompress_egd, load_and_decompress_igd,
    load_and_decompress_tgd,
};
pub use image_processor::{BuildImageBitDataSet, ImageColorSpace, OpenImage};
pub use tabular_processor::{
    BuildBitDataSet, DEFAULT_DECIMAL_SCALE, FloatScalingMode, PreprocessOptions, ReconstructDataFrame,
    reconstruct_to_dataframe,
};
