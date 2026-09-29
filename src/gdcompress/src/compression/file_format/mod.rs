mod egd;
mod igd;
mod path_utils;
pub(crate) mod tags;
mod tgd;

pub use egd::{
    DecodeDeltaBaseTable, EgdFile, FORMAT_VERSION, LoadEgdFile, MAGIC_BYTES, SaveEgdFile,
    load_and_decompress_egd, load_compressed_from_egd, save_compressed_as_egd,
};
pub use igd::{
    IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES, IgdFile, LoadIgdFile, SaveIgdFile,
    decompress_igd_to_image, load_and_decompress_igd, load_compressed_from_igd,
    save_compressed_as_igd,
};
pub use tgd::{
    TGD_FORMAT_VERSION, TGD_MAGIC_BYTES, LoadTgdFile, SaveTgdFile, TgdFile,
    decompress_tgd_to_csv, load_and_decompress_tgd, load_compressed_from_tgd,
    save_compressed_as_tgd,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodeDataOffsetRLE;
    use crate::compression::base_selection::SelectBases;
    use crate::compression::base_table::BuildBaseTable;
    use crate::compression::condensed_samples::GenCondensedSamples;
    use crate::compression::data::{
        BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, FeatureDataType, FeatureSpec,
        FeatureTransform, ImageColorModel, ImageGroupingTransform, ImageReconstructionInfo,
        PixelGrouping,
    };
    use crate::compression::decompression::decompress_file;
    use crate::compression::encoding::{CompressedData, EncodedData};
    use crate::compression::encoding::{EncodeData, EncodeDataHuffman};
    use crate::compression::entropy::Entropy;
    use crate::filter_pipeline::{Filter, FilterExt};
    use bitvec::prelude::*;
    use polars::prelude::DataType;

    fn get_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        Entropy {}
            .then(GenCondensedSamples { m_max: 50 })
            .then(SelectBases { patience: 10 })
            .then(BuildBaseTable {})
            .then(EncodeData {})
    }

    fn get_rle_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        Entropy {}
            .then(GenCondensedSamples { m_max: 100 })
            .then(SelectBases { patience: 5 })
            .then(BuildBaseTable {})
            .then(EncodeDataOffsetRLE {})
    }

    fn get_huffman_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData>
    {
        Entropy {}
            .then(GenCondensedSamples { m_max: 100 })
            .then(SelectBases { patience: 5 })
            .then(BuildBaseTable {})
            .then(EncodeDataHuffman {})
    }

    #[test]
    fn test_build_and_save_egd() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 64],
            num_rows: 1,
            stride: 64,
            chunk_size: 64,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(8),
                transform: FeatureTransform::None
            };
            8
        ];
        let info = BitDataInfo::new(features, 64).unwrap();
        let bit_data = BitDataSet { data, info };
        let pipeline = get_compression_pipeline();
        let compressed = pipeline.process(bit_data).unwrap();
        let egd = EgdFile::from_compressed_data(compressed.clone()).unwrap();

        assert!(egd.as_bytes().len() >= 4);
        assert_eq!(&egd.as_bytes()[0..3], &MAGIC_BYTES);
        assert_eq!(egd.as_bytes()[3], FORMAT_VERSION);

        let output = std::env::temp_dir().join("gdcompress_test_output");
        let saved = egd.save(&output).unwrap();
        assert_eq!(saved.extension().and_then(|s| s.to_str()), Some("egd"));
        assert!(saved.exists());
        let _ = std::fs::remove_file(saved);
    }

    #[test]
    fn test_roundtrip_egd_to_compressed_data() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 320],
            num_rows: 5,
            stride: 64,
            chunk_size: 64,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(32),
                transform: FeatureTransform::None,
            },
            FeatureSpec {
                data_type: FeatureDataType::UInt(32),
                transform: FeatureTransform::None,
            },
        ];
        let info = BitDataInfo::new(features, 320).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let egd = EgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata, compressed.metadata);
        assert_eq!(
            loaded.layout.selected_base_bit_positions(),
            compressed.layout.selected_base_bit_positions()
        );
        assert_eq!(
            loaded.layout.variable_base_bit_positions(),
            compressed.layout.variable_base_bit_positions()
        );
        assert_eq!(
            loaded.layout.constant_zero_bit_positions(),
            compressed.layout.constant_zero_bit_positions()
        );
        assert_eq!(
            loaded.layout.constant_one_bit_positions(),
            compressed.layout.constant_one_bit_positions()
        );
        assert_eq!(
            loaded.condensed_sample_weights,
            compressed.condensed_sample_weights
        );
        assert_eq!(
            loaded.encoded_data.encoded_bit_stream(),
            compressed.encoded_data.encoded_bit_stream()
        );
        assert_eq!(
            loaded.encoded_data.get_num_samples(),
            compressed.encoded_data.get_num_samples()
        );
        assert_eq!(
            loaded.encoded_data.get_num_deviation_bits(),
            compressed.encoded_data.get_num_deviation_bits()
        );
        assert_eq!(
            loaded.encoded_data.get_num_id_bits(),
            compressed.encoded_data.get_num_id_bits()
        );
        assert_eq!(loaded.base_table.len(), compressed.base_table.len());
        for (lhs, rhs) in loaded
            .base_table
            .as_raw()
            .iter()
            .zip(compressed.base_table.as_raw().iter())
        {
            assert_eq!(lhs.0, rhs.0);
            assert_eq!(lhs.1, rhs.1);
        }
    }

    #[test]
    fn test_roundtrip_igd_to_compressed_data_with_image_metadata() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 96],
            num_rows: 4,
            stride: 24,
            chunk_size: 24,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(8),
                transform: FeatureTransform::None
            };
            3
        ];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            96,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(1, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let igd = IgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata.num_features(), 3);
        assert_eq!(loaded.metadata.original_size_bits(), 96);
        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(1, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));
    }

    #[test]
    fn test_roundtrip_igd_to_compressed_data_with_for_min_metadata() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 111],
            num_rows: 1,
            stride: 111,
            chunk_size: 111,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(37),
                transform: FeatureTransform::None
            };
            3
        ];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            111,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping::new(1, 1),
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let igd = IgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(1, 1),
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0
            })
        ));
    }

    #[test]
    fn test_roundtrip_egd_with_rle_payload() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 512],
            num_rows: 8,
            stride: 64,
            chunk_size: 64,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(8),
                transform: FeatureTransform::None
            };
            8
        ];
        let info = BitDataInfo::new(features, 512).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_rle_compression_pipeline().process(bit_data).unwrap();
        let egd = EgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        match (&compressed.encoded_data, &loaded.encoded_data) {
            (EncodedData::RleOffset(src), EncodedData::RleOffset(dst)) => {
                let src_as_raw = src.to_deviation_data().unwrap();
                let dst_as_raw = dst.to_deviation_data().unwrap();
                assert_eq!(
                    src_as_raw.encoded_bit_stream(),
                    dst_as_raw.encoded_bit_stream()
                );
                assert_eq!(src_as_raw.get_num_samples(), dst_as_raw.get_num_samples());
                assert_eq!(
                    src_as_raw.get_num_deviation_bits(),
                    dst_as_raw.get_num_deviation_bits()
                );
                assert_eq!(src_as_raw.get_num_id_bits(), dst.get_num_id_bits());
            }
            _ => panic!("expected source RLE data and loaded normalized data"),
        }
    }

    #[test]
    fn test_roundtrip_egd_with_huffman_payload() {
        let data = BitData {
            data: vec![
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                1, 1, 0, 0,
            ]
            .into_iter()
            .map(|b| b != 0)
            .collect(),
            num_rows: 4,
            stride: 8,
            chunk_size: 8,
        };
        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UInt(8),
            transform: FeatureTransform::None,
        }];
        let info = BitDataInfo::new(features, 32).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_huffman_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let egd = EgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        assert!(matches!(loaded.encoded_data, EncodedData::Huffman(_)));
        assert_eq!(loaded.metadata, compressed.metadata);
        let decompressed_data = decompress_file(loaded).unwrap().data;
        assert_eq!(decompressed_data.num_rows, bit_data.data.num_rows);
        assert_eq!(decompressed_data.chunk_size, bit_data.data.chunk_size);
        for row in 0..bit_data.data.num_rows {
            assert_eq!(
                decompressed_data.get_chunk(row),
                bit_data.data.get_chunk(row)
            );
        }
    }

    #[test]
    fn test_roundtrip_igd_with_huffman_image_payload() {
        let data = BitData {
            data: vec![
                0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
            ]
            .into_iter()
            .map(|b| b != 0)
            .collect(),
            num_rows: 2,
            stride: 12,
            chunk_size: 12,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(4),
                transform: FeatureTransform::None
            };
            3
        ];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            24,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping::new(4, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_huffman_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let igd = IgdFile::from_compressed_data(compressed.clone()).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert!(matches!(loaded.encoded_data, EncodedData::Huffman(_)));
        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: PixelGrouping(4, 1),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));
        let decompressed_data = decompress_file(loaded).unwrap().data;
        assert_eq!(decompressed_data.num_rows, bit_data.data.num_rows);
        assert_eq!(decompressed_data.chunk_size, bit_data.data.chunk_size);
        for row in 0..bit_data.data.num_rows {
            assert_eq!(
                decompressed_data.get_chunk(row),
                bit_data.data.get_chunk(row)
            );
        }
    }

    #[test]
    fn test_tgd_preserves_long_column_names() {
        let data = BitData {
            data: bitvec![usize, Lsb0; 0; 384],
            num_rows: 6,
            stride: 64,
            chunk_size: 64,
        };
        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UInt(8),
            transform: FeatureTransform::None,
        }; 8];

        let column_names = vec![
            "a_very_long_column_name_that_exceeds_typical_lengths_123456".to_string(),
            "b".to_string(),
            "another_extremely_verbose_header_describing_an_obscure_metric_abc".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
            "f".to_string(),
            "this_is_the_eighth_column_with_a_particularly_verbose_name_xyz".to_string(),
        ];
        let original_dtypes = vec![
            DataType::Float64,
            DataType::Int32,
            DataType::UInt8,
            DataType::Float32,
            DataType::Int64,
            DataType::UInt16,
            DataType::Float16,
            DataType::UInt64,
        ];

        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            6 * 64,  // num_rows * chunk_size
            BitDataReconstructionInfo::Tabular {
                column_names: column_names.clone(),
                original_dtypes: original_dtypes.clone(),
            },
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();
        let tgd = TgdFile::from_compressed_data(compressed).unwrap();
        let loaded = tgd.to_compressed_data().unwrap();

        match &loaded.metadata.reconstruction {
            BitDataReconstructionInfo::Tabular {
                column_names: loaded_names,
                original_dtypes: loaded_dtypes,
            } => {
                assert_eq!(*loaded_names, column_names);
                assert_eq!(*loaded_dtypes, original_dtypes);
            }
            _ => panic!("expected Tabular reconstruction metadata"),
        }
    }
}
