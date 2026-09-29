use gdcompress::compression::BaseTable;
use gdcompress::load_csv;
use gdcompress::prelude::*;
use gdcompress::{BitDataSet, DecompressFileData, LoadEgdFile, PreprocessOptions, SaveEgdFile};
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

/// Test roundtrip compression with delta-encoded base table using unary prefix
#[test]
fn test_delta_codec_roundtrip_unary_prefix() {
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
        .then(BuildSortedBaseTable {})
        .then(EncodeData {})
        .then(DeltaEncodeBaseTable {}); // Unary prefix codec

    let compressed = compression_pipeline
        .process(bit_data.clone())
        .expect("Failed to compress with unary delta codec");

    // Verify that the base table is delta-encoded
    assert!(
        matches!(&compressed.base_table, BaseTable::Delta(_),),
        "Expected delta-encoded base table"
    );

    let temp_egd_path = env::temp_dir().join("test_delta_unary.egd");

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

    // Verify the decompressed data matches the original
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

    assert_bitstream_eq(
        bit_data.data.data.as_bitslice(),
        decompressed.data.data.as_bitslice(),
        "Unary codec: Decompressed BitData content does not match original",
    );

    // Cleanup
    if temp_egd_path.exists() {
        std::fs::remove_file(temp_egd_path).ok();
    }
}

/// Test roundtrip compression with delta-encoded base table using fixed prefix
#[test]
fn test_delta_codec_roundtrip_fixed_prefix() {
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
        .then(BuildSortedBaseTable {})
        .then(EncodeData {})
        .then(DeltaEncodeBaseTableFixed {}); // Fixed 4-bit prefix codec

    let compressed = compression_pipeline
        .process(bit_data.clone())
        .expect("Failed to compress with fixed prefix delta codec");

    // Verify that the base table is delta-encoded
    assert!(
        matches!(&compressed.base_table, BaseTable::Delta(_),),
        "Expected delta-encoded base table"
    );

    let temp_egd_path = env::temp_dir().join("test_delta_fixed.egd");

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

    // Verify the decompressed data matches the original
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

    assert_bitstream_eq(
        bit_data.data.data.as_bitslice(),
        decompressed.data.data.as_bitslice(),
        "Fixed codec: Decompressed BitData content does not match original",
    );

    // Cleanup
    if temp_egd_path.exists() {
        std::fs::remove_file(temp_egd_path).ok();
    }
}

/// Test that both codecs can be read (codec tag is preserved correctly)
#[test]
fn test_delta_codec_tag_serialization() {
    let input_path = "data/tabular/data-10000-8-int.csv";
    let loaded = load_csv(input_path, true, None).expect("Failed to load CSV");

    let bit_data = BitDataSet::from_dataframe(loaded, PreprocessOptions::default(), false)
        .expect("Failed to create BitDataSet");

    // Test unary prefix codec serialization
    let compression_pipeline_unary = Entropy {}
        .then(GenCondensedSamples { m_max: 50 })
        .then(SelectBasesThreshold {
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
            entropy_threshold: 0.70,
        })
        .then(BuildSortedBaseTable {})
        .then(EncodeData {})
        .then(DeltaEncodeBaseTable {});

    let compressed_unary = compression_pipeline_unary
        .process(bit_data.clone())
        .expect("Failed to compress with unary codec");

    // Extract codec_id from the delta base table
    let unary_codec_id = if let BaseTable::Delta(delta) = &compressed_unary.base_table {
        delta.codec_id
    } else {
        panic!("Expected delta-encoded base table");
    };

    assert_eq!(
        unary_codec_id, 1,
        "Unary codec should have codec_id = 1 (BASE_TABLE_TAG_DELTA_UNARY)"
    );

    // Test fixed prefix codec serialization
    let compression_pipeline_fixed = Entropy {}
        .then(GenCondensedSamples { m_max: 50 })
        .then(SelectBasesThreshold {
            patience: 10,
            base_bit_impl: BaseBitImpl::Naive,
            entropy_threshold: 0.70,
        })
        .then(BuildSortedBaseTable {})
        .then(EncodeData {})
        .then(DeltaEncodeBaseTableFixed {});

    let compressed_fixed = compression_pipeline_fixed
        .process(bit_data.clone())
        .expect("Failed to compress with fixed codec");

    // Extract codec_id from the delta base table
    let fixed_codec_id = if let BaseTable::Delta(delta) = &compressed_fixed.base_table {
        delta.codec_id
    } else {
        panic!("Expected delta-encoded base table");
    };

    assert_eq!(
        fixed_codec_id, 2,
        "Fixed codec should have codec_id = 2 (BASE_TABLE_TAG_DELTA_FIXED)"
    );

    // Verify codecs are different
    assert_ne!(
        unary_codec_id, fixed_codec_id,
        "Unary and fixed codecs should have different codec_ids"
    );
}
