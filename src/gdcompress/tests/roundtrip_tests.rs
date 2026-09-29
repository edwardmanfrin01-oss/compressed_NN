use gdcompress::load_csv;
use gdcompress::prelude::*;
use gdcompress::{
    BitDataSet, DecompressFileData, EncodedData, ImageColorModel, LoadEgdFile, PreprocessOptions,
    SaveEgdFile, SaveIgdFile, decompress_igd_to_image,
};
use std::env;

fn assert_bitstream_eq(
    expected: &bitvec::slice::BitSlice,
    actual: &bitvec::slice::BitSlice,
    context: &str,
) {
    if expected == actual {
        return;
    }

    let mismatch_idx = expected
        .iter()
        .zip(actual.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| expected.len().min(actual.len()));

    let expected_bit = expected.get(mismatch_idx).map(|b| *b);
    let actual_bit = actual.get(mismatch_idx).map(|b| *b);

    panic!(
        "{}: first mismatch at bit {}, expected={:?}, actual={:?}, expected_len={}, actual_len={}",
        context,
        mismatch_idx,
        expected_bit,
        actual_bit,
        expected.len(),
        actual.len()
    );
}

fn assert_chunk_data_eq(expected: &BitDataSet, actual: &BitDataSet, context: &str) {
    assert_eq!(
        expected.data.num_rows, actual.data.num_rows,
        "{}: num_rows mismatch",
        context
    );
    assert_eq!(
        expected.data.chunk_size, actual.data.chunk_size,
        "{}: chunk_size mismatch",
        context
    );

    for row in 0..expected.data.num_rows {
        let expected_chunk = expected.data.get_chunk(row);
        let actual_chunk = actual.data.get_chunk(row);
        if expected_chunk != actual_chunk {
            panic!("{}: first mismatching row {}", context, row);
        }
    }
}

#[test]
fn test_csv_roundtrip_compression() {
    let input_path = "data/tabular/data-10000-8-int.csv";
    let loaded = load_csv(input_path, true, None).expect("Failed to load CSV");

    let bit_data = BitDataSet::from_dataframe(loaded, PreprocessOptions::default(), false)
        .expect("Failed to create BitDataSet");

    let compression_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 50 })
        .then(SelectBasesThreshold {
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
            entropy_threshold: 0.70,
        })
        .then(BuildBaseTable {})
        .then(EncodeData {});

    let compressed = compression_pipeline
        .process(bit_data.clone())
        .expect("Failed to compress CSV");

    let temp_egd_path = env::temp_dir().join("test_roundtrip.egd");

    let saved_path = SaveEgdFile {
        output_path: temp_egd_path.clone(),
    }
    .process(compressed)
    .expect("Failed to save EGD file");

    let loaded_compressed = LoadEgdFile {}
        .process(saved_path)
        .expect("Failed to load EGD file");

    let decompressed = DecompressFileData {}
        .process(loaded_compressed)
        .expect("Failed to decompress data");

    assert_eq!(
        bit_data.data.total_bits(),
        decompressed.data.total_bits(),
        "Decompressed BitData total bits does not match original"
    );
    assert_eq!(
        bit_data.data.chunk_size, decompressed.data.chunk_size,
        "Decompressed BitData chunk size does not match original"
    );
    assert_eq!(
        bit_data.data.num_rows, decompressed.data.num_rows,
        "Decompressed BitData num rows does not match original"
    );

    // Deep data verification: Check that every single bit is identical
    assert_bitstream_eq(
        bit_data.data.data.as_bitslice(),
        decompressed.data.data.as_bitslice(),
        "Decompressed BitData content does not perfectly match original",
    );

    // Cleanup
    if temp_egd_path.exists() {
        std::fs::remove_file(temp_egd_path).ok();
    }
}

#[test]
fn test_csv_roundtrip_compression_rle() {
    let input_path = "data/tabular/data-10000-8-int.csv";
    let loaded = load_csv(input_path, true, None).expect("Failed to load CSV");

    let bit_data = BitDataSet::from_dataframe(loaded, PreprocessOptions::default(), false)
        .expect("Failed to create BitDataSet");

    let compression_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 50 })
        .then(SelectBasesThreshold {
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
            entropy_threshold: 0.70,
        })
        .then(BuildBaseTable {})
        .then(EncodeDataOffsetRLE {});

    let compressed = compression_pipeline
        .process(bit_data.clone())
        .expect("Failed to compress CSV with RLE");

    let temp_egd_path = env::temp_dir().join("test_roundtrip_rle.egd");

    let saved_path = SaveEgdFile {
        output_path: temp_egd_path.clone(),
    }
    .process(compressed)
    .expect("Failed to save EGD file");

    let loaded_compressed = LoadEgdFile {}
        .process(saved_path)
        .expect("Failed to load EGD file");

    let decompressed = DecompressFileData {}
        .process(loaded_compressed)
        .expect("Failed to decompress data");

    assert_eq!(
        bit_data.data.total_bits(),
        decompressed.data.total_bits(),
        "Decompressed BitData total bits does not match original"
    );
    assert_eq!(
        bit_data.data.chunk_size, decompressed.data.chunk_size,
        "Decompressed BitData chunk size does not match original"
    );
    assert_eq!(
        bit_data.data.num_rows, decompressed.data.num_rows,
        "Decompressed BitData num rows does not match original"
    );
    assert_eq!(
        bit_data.data.data, decompressed.data.data,
        "Decompressed BitData content does not perfectly match original"
    );

    if temp_egd_path.exists() {
        std::fs::remove_file(temp_egd_path).ok();
    }
}

#[test]
fn test_image_roundtrip_compression() {
    let input_path = std::path::PathBuf::from("data/images/rustacean.png");
    assert!(input_path.exists(), "Image file not found");

    let compression_pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: PixelGrouping::new(3, 1),
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: true,
    }
    .then(Entropy {})
    .then(GenCondensedSamples { m_max: 0 })
    .then(SelectBasesThreshold {
        patience: 10,
        base_bit_impl: BaseBitImpl::Naive,
        entropy_threshold: 0.70,
    })
    .then(BuildBaseTable {})
    .then(EncodeDataHuffman {});

    let compressed = compression_pipeline
        .process(OpenImage.process(input_path.clone()).unwrap())
        .expect("Failed to compress image");

    let temp_igd_path = env::temp_dir().join("test_rustacean.igd");
    let temp_png_path = env::temp_dir().join("test_rustacean_out.png");

    let saved_path = SaveIgdFile {
        output_path: temp_igd_path.clone(),
    }
    .process(compressed)
    .expect("Failed to save IGD file");

    decompress_igd_to_image(&saved_path, &temp_png_path).expect("Failed to decompress image");

    assert!(temp_png_path.exists(), "Output image was not created");

    let metadata = std::fs::metadata(&temp_png_path).expect("Failed to get metadata");
    assert!(metadata.len() > 0, "Output image is empty");

    // Exact Image Verification
    let original_rgba = image::open(&input_path)
        .expect("Failed to open original image")
        .into_rgba8();
    let decompressed_rgba = image::open(&temp_png_path)
        .expect("Failed to open decompressed image")
        .into_rgba8();

    assert_eq!(
        original_rgba.into_raw(),
        decompressed_rgba.into_raw(),
        "Decompressed image pixels do not perfectly match original"
    );

    // Cleanup
    if temp_igd_path.exists() {
        std::fs::remove_file(temp_igd_path).ok();
    }
    if temp_png_path.exists() {
        std::fs::remove_file(temp_png_path).ok();
    }
}

#[test]
fn test_image_roundtrip_compression_rle_offset() {
    let input_path = std::path::PathBuf::from("data/images/rustacean.png");
    assert!(input_path.exists(), "Image file not found");

    let bit_data = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: PixelGrouping::new(3, 1),
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: true,
    }
    .process(OpenImage.process(input_path).unwrap())
    .expect("Failed to build image BitDataSet");

    let compression_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 0 })
        .then(SelectBasesThreshold {
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
            entropy_threshold: 0.70,
        })
        .then(BuildBaseTable {})
        .then(EncodeDataOffsetRLE {});

    let compressed = compression_pipeline
        .process(bit_data.clone())
        .expect("Failed to compress image with RLE row offsets");

    match &compressed.encoded_data {
        EncodedData::RleOffset(data) => {
            assert!(
                !data.row_offsets().is_empty(),
                "RLE row offsets should be present for image inputs"
            );
            assert_eq!(data.row_offsets()[0], (0, 0));
        }
        _ => panic!("Expected RLE offset encoded data"),
    }

    let pre_save_decompressed = DecompressFileData {}
        .process(compressed.clone())
        .expect("Failed to decompress in-memory RLE offset data");
    assert_chunk_data_eq(
        &bit_data,
        &pre_save_decompressed,
        "In-memory RLE offset decompression does not match original",
    );

    let temp_egd_path = env::temp_dir().join("test_roundtrip_rle_offset.egd");
    let saved_path = SaveEgdFile {
        output_path: temp_egd_path.clone(),
    }
    .process(compressed)
    .expect("Failed to save EGD file");

    let loaded_compressed = LoadEgdFile {}
        .process(saved_path)
        .expect("Failed to load EGD file");
    assert!(matches!(
        loaded_compressed.encoded_data,
        EncodedData::RleOffset(_)
    ));

    let decompressed = DecompressFileData {}
        .process(loaded_compressed)
        .expect("Failed to decompress RLE offset data");

    assert_eq!(
        bit_data.data.total_bits(),
        decompressed.data.total_bits(),
        "Decompressed BitData total bits does not match original"
    );
    assert_eq!(
        bit_data.data.chunk_size, decompressed.data.chunk_size,
        "Decompressed BitData chunk size does not match original"
    );
    assert_eq!(
        bit_data.data.num_rows, decompressed.data.num_rows,
        "Decompressed BitData num rows does not match original"
    );
    assert_chunk_data_eq(
        &bit_data,
        &decompressed,
        "EGD-loaded RLE offset decompression does not match original",
    );

    if temp_egd_path.exists() {
        std::fs::remove_file(temp_egd_path).ok();
    }
}
