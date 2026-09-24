use gdcompress::compression::base_bits::BaseBitGroups;
use gdcompress::prelude::*;
use gdcompress::{BaseSelectionContext, BitDataSet, IgdFile, PreEncodeContext};
use igd_extract::{Options, extract, reconstruct_chunk};
use image::{DynamicImage, Rgb, RgbImage};
use serde_json::Value;

fn fixture(transform: ImageGroupingTransform, constant: bool) -> (BitDataSet, PreEncodeContext) {
    // Odd dimensions exercise partial edge groups. Restricted channel ranges
    // create both constant-zero and constant-one bits in the raw representation.
    // No partial groups for the constant-only case: padding at image borders
    // would introduce zero values and make an otherwise solid image variable.
    let (width, height) = if constant { (12, 9) } else { (11, 8) };
    let image = RgbImage::from_fn(width, height, |x, y| {
        if constant {
            Rgb([255, 0, 128])
        } else {
            Rgb([
                128 | ((x / 2 + 3 * y) % 8) as u8,
                ((5 * x + y / 2) % 16) as u8,
                64 | ((x + y) % 4) as u8,
            ])
        }
    });
    let data = BuildImageBitDataSet {
        color_model: if matches!(transform, ImageGroupingTransform::Raw) {
            ImageColorModel::Rgb
        } else {
            ImageColorModel::YCoCgR
        },
        pixel_grouping: PixelGrouping::new(3, 3),
        grouping_transform: transform,
        ..Default::default()
    }
    .process(DynamicImage::ImageRgb8(image))
    .unwrap();
    let scored = Entropy.process(data.clone()).unwrap();
    let mut groups = BaseBitGroups::new(data.num_rows(), data.chunk_size());
    let constants: Vec<_> = scored
        .entropy_scores
        .iter()
        .filter(|(_, h)| *h == 0.0)
        .map(|(p, _)| *p)
        .collect();
    groups.add_constant_bit_positions(&constants);
    // Deliberately non-ascending selection order: JSON must restore the positional
    // mapping even after dictionary sorting and delta column permutation.
    for &(position, _) in scored
        .entropy_scores
        .iter()
        .rev()
        .filter(|(_, h)| *h != 0.0)
        .step_by(3)
        .take(6)
    {
        groups.add_bit_position(&data, position);
    }
    let context = BuildSortedBaseTable {}
        .process(BaseSelectionContext::new(
            data.clone(),
            Box::new(groups),
            scored.constant_bit_polarity,
        ))
        .unwrap();
    (data, context)
}

fn encode(context: PreEncodeContext, encoding: usize, dictionary: usize) -> Vec<u8> {
    let compressed = match encoding {
        0 => EncodeData {}.process(context).unwrap(),
        1 => EncodeDataOffsetRLE {}.process(context).unwrap(),
        _ => EncodeDataHuffman {}.process(context).unwrap(),
    };
    let compressed = match dictionary {
        0 => compressed,
        1 => DeltaEncodeBaseTable {}.process(compressed).unwrap(),
        _ => DeltaEncodeBaseTableFixed {}.process(compressed).unwrap(),
    };
    IgdFile::from_compressed_data(compressed)
        .unwrap()
        .as_bytes()
        .to_vec()
}

#[test]
fn serialized_export_restores_input_bits_for_all_codecs_and_transforms() {
    for transform in [
        ImageGroupingTransform::Raw,
        ImageGroupingTransform::ForFirstPixel,
        ImageGroupingTransform::ForMin,
    ] {
        let (original, context) = fixture(transform, false);
        for encoding in 0..3 {
            for dictionary in 0..3 {
                let bytes = encode(context.clone(), encoding, dictionary);
                let document = extract(
                    bytes,
                    "fixture.igd",
                    Options {
                        include_deviations: true,
                        verify: true,
                    },
                )
                .unwrap();
                // Validate the serialized JSON, not just internal Rust structures.
                let document: Value =
                    serde_json::from_slice(&serde_json::to_vec_pretty(&document).unwrap()).unwrap();
                assert_eq!(
                    document["sample_base_ids"],
                    serde_json::json!(context.row_to_base_id)
                );
                assert!(document["selection_order"].is_null());
                assert_eq!(document["verification"]["passed"], true);
                assert_eq!(document["image"]["grid_width"], 4);
                assert_eq!(document["image"]["grid_height"], 3);
                for i in 0..original.num_rows() {
                    let rebuilt = reconstruct_chunk(&document, i).unwrap();
                    let start = i * original.data.stride;
                    assert_eq!(
                        rebuilt.as_bitslice(),
                        &original.data.data[start..start + original.chunk_size()],
                        "sample {i}, {transform:?}, encoding {encoding}, dictionary {dictionary}"
                    );
                }
                let frequencies: u64 = document["bases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| b["frequency"].as_u64().unwrap())
                    .sum();
                assert_eq!(frequencies, original.num_rows() as u64);
            }
        }
    }
}

#[test]
fn constant_only_dictionary_and_optional_deviations() {
    let (_, context) = fixture(ImageGroupingTransform::Raw, true);
    assert!(!context.layout.constant_zero_bit_positions().is_empty());
    assert!(!context.layout.constant_one_bit_positions().is_empty());
    assert!(context.layout.variable_base_bit_positions().is_empty());
    for encoding in 0..3 {
        for dictionary in 0..3 {
            let bytes = encode(context.clone(), encoding, dictionary);
            let document = extract(
                bytes,
                "solid.igd",
                Options {
                    include_deviations: false,
                    verify: true,
                },
            )
            .unwrap();
            assert_eq!(document["num_bases"], 1);
            assert_eq!(document["base_bits"], 216);
            assert_eq!(document["variable_base_bits"], 0);
            assert_eq!(document["bases"][0]["variable_bits"], "");
            assert_eq!(document["bases"][0]["frequency"], 12);
            assert!(document["sample_deviation_bits"].is_null());
            assert_eq!(document["verification"]["passed"], true);
        }
    }
}

#[test]
fn rejects_truncated_input_and_inconsistent_export_mapping() {
    let (_, context) = fixture(ImageGroupingTransform::ForFirstPixel, false);
    let bytes = encode(context, 2, 2);
    for len in [0, 3, 31, 40, bytes.len() - 1] {
        assert!(extract(bytes[..len].to_vec(), "truncated.igd", Options::default()).is_err());
    }
    let mut document = extract(
        bytes,
        "fixture.igd",
        Options {
            include_deviations: true,
            verify: false,
        },
    )
    .unwrap();
    document["base_bit_positions"][1] = document["base_bit_positions"][0].clone();
    assert!(reconstruct_chunk(&document, 0).is_err());
}

#[test]
fn command_writes_valid_json_and_refuses_overwrite_or_truncated_input() {
    let (_, context) = fixture(ImageGroupingTransform::ForFirstPixel, false);
    let bytes = encode(context, 2, 2);
    let folder = std::env::temp_dir().join(format!(
        "igd-extract-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&folder).unwrap();
    let input = folder.join("input.igd");
    let output = folder.join("output.json");
    std::fs::write(&input, &bytes).unwrap();
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_igd_extract"))
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .arg("--verify")
            .arg("--include-deviations")
            .output()
            .unwrap()
    };
    let first = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let saved = std::fs::read(&output).unwrap();
    let parsed: Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(parsed["verification"]["passed"], true);
    assert!(!run().status.success());
    assert_eq!(std::fs::read(&output).unwrap(), saved);
    std::fs::remove_file(&output).unwrap();
    std::fs::write(&input, &bytes[..bytes.len() - 1]).unwrap();
    assert!(!run().status.success());
    assert!(!output.exists());
    std::fs::remove_file(&input).unwrap();
    std::fs::remove_dir(&folder).unwrap();
}
