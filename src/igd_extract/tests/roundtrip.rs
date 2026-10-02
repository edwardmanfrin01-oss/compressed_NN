use gdcompress::compression::base_bits::BaseBitGroups;
use gdcompress::prelude::*;
use gdcompress::{BaseSelectionContext, BitDataSet, IgdFile, PreEncodeContext};
use igd_extract::{
    Options, extract, extract_network_features, extract_network_u8, reconstruct_chunk,
    write_network_npz,
};
use image::{DynamicImage, Rgb, RgbImage};
use serde_json::Value;

fn fixture(transform: ImageGroupingTransform, constant: bool) -> (BitDataSet, PreEncodeContext) {
    fixture_grouped(transform, constant, PixelGrouping::new(3, 3))
}

fn fixture_grouped(
    transform: ImageGroupingTransform,
    constant: bool,
    grouping: PixelGrouping,
) -> (BitDataSet, PreEncodeContext) {
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
        pixel_grouping: grouping,
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
fn cumulative_maps_original_ids_and_preserves_constant_case() {
    use igd_extract::extract_cumulative_u16;
    for constant in [false, true] {
        let (_, context) = fixture_grouped(
            ImageGroupingTransform::Raw,
            constant,
            PixelGrouping::new(1, 1),
        );
        for encoding in 0..3 {
            for codec in 0..3 {
                let bytes = encode(context.clone(), encoding, codec);
                let doc = extract(bytes.clone(), "fixture", Options::default()).unwrap();
                let mut bases: Vec<_> = doc["bases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| {
                        (
                            u64::from_str_radix(b["bits"].as_str().unwrap(), 2).unwrap(),
                            b["id"].as_u64().unwrap() as usize,
                        )
                    })
                    .collect();
                bases.sort();
                let gaps: Vec<_> = bases.windows(2).map(|p| p[1].0 - p[0].0).collect();
                let max = *gaps.iter().max().unwrap_or(&0);
                for scale in [0, 1, 3] {
                    let mut lookup = vec![0u16; bases.len()];
                    let mut cumulative = 0u16;
                    for (r, &(_, id)) in bases.iter().enumerate() {
                        if r > 0 {
                            cumulative += (f64::from(scale) * ((gaps[r - 1] + 1) as f64).log2()
                                / ((max + 1) as f64).log2())
                            .round() as u16;
                        }
                        lookup[id] = r as u16 + cumulative;
                    }
                    let result = extract_cumulative_u16(
                        bytes.clone(),
                        "fixture",
                        Options {
                            verify: true,
                            include_deviations: false,
                        },
                        scale,
                    )
                    .unwrap();
                    let expected: Vec<_> = context
                        .row_to_base_id
                        .iter()
                        .map(|id| lookup[*id])
                        .collect();
                    assert_eq!(result.z_u16, expected);
                    if constant {
                        assert!(result.z_u16.iter().all(|v| *v == 0));
                    }
                }
            }
        }
    }
    let (_, grouped) = fixture(ImageGroupingTransform::Raw, false);
    assert!(
        extract_cumulative_u16(encode(grouped, 0, 0), "grouped", Options::default(), 1).is_err()
    );
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
fn network_representation_sorts_bases_and_preserves_spatial_dictionary_links() {
    let (_, context) = fixture(ImageGroupingTransform::Raw, false);
    let bytes = encode(context.clone(), 1, 2);
    let raw = extract(bytes.clone(), "fixture.igd", Options::default()).unwrap();
    let features = extract_network_features(
        bytes,
        "fixture.igd",
        Options {
            include_deviations: false,
            verify: true,
        },
    )
    .unwrap();
    assert_eq!(features["schema"], "igd-network-features-v1");
    assert_eq!(features["image"]["grid_width"], 4);
    assert_eq!(features["image"]["grid_height"], 3);
    assert_eq!(features["verification"]["passed"], true);
    let dictionary = features["sorted_dictionary"].as_array().unwrap();
    assert_eq!(
        dictionary.len(),
        raw["num_bases"].as_u64().unwrap() as usize
    );
    let mut old_to_rank = vec![0usize; dictionary.len()];
    for (rank, entry) in dictionary.iter().enumerate() {
        assert_eq!(entry["rank"].as_u64().unwrap() as usize, rank);
        old_to_rank[entry["original_id"].as_u64().unwrap() as usize] = rank;
        if rank == 0 {
            assert!(
                entry["delta_bits"]
                    .as_str()
                    .unwrap()
                    .bytes()
                    .all(|bit| bit == b'0')
            );
        } else {
            assert!(
                entry["base_bits"].as_str().unwrap()
                    > dictionary[rank - 1]["base_bits"].as_str().unwrap()
            );
        }
    }
    let ranks: Vec<usize> = features["spatial_rank_ids"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|rank| rank.as_u64().unwrap() as usize)
        .collect();
    let old_ids: Vec<usize> = raw["sample_base_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as usize)
        .collect();
    let expected: Vec<usize> = old_ids.iter().map(|id| old_to_rank[*id]).collect();
    assert_eq!(ranks, expected);
    let frequency: u64 = dictionary
        .iter()
        .map(|entry| entry["frequency"].as_u64().unwrap())
        .sum();
    assert_eq!(frequency, old_ids.len() as u64);
    let delta_map = features["spatial_delta_bits"].as_array().unwrap();
    for (sample, delta) in delta_map
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .enumerate()
    {
        assert_eq!(delta, &dictionary[ranks[sample]]["delta_bits"]);
    }
}

#[test]
fn network_u8_quantizes_two_maps_and_writes_npz() {
    let (_, context) = fixture(ImageGroupingTransform::Raw, false);
    let bytes = encode(context, 2, 2);
    let features =
        extract_network_features(bytes.clone(), "fixture.igd", Options::default()).unwrap();
    let network = extract_network_u8(bytes.clone(), "fixture.igd", Options::default()).unwrap();
    assert_eq!((network.width, network.height), (4, 3));
    assert_eq!(network.rank_u8.len(), 12);
    assert_eq!(network.delta_u8.len(), 12);
    for ((rank, delta), (rank_u8, delta_u8)) in features["spatial_rank_normalized"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .zip(
            features["spatial_delta_log2_normalized"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter()),
        )
        .zip(network.rank_u8.iter().zip(network.delta_u8.iter()))
    {
        assert_eq!(*rank_u8, (rank.as_f64().unwrap() * 255.0).round() as u8);
        assert_eq!(*delta_u8, (delta.as_f64().unwrap() * 255.0).round() as u8);
    }
    let folder = std::env::temp_dir().join(format!(
        "igd-extract-npz-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&folder).unwrap();
    let output = folder.join("features.npz");
    let summary = write_network_npz(bytes, "fixture.igd", Options::default(), &output).unwrap();
    assert_eq!((summary.width, summary.height), (4, 3));
    assert_eq!(summary.tensor_bytes, 24);
    let archive = std::fs::read(&output).unwrap();
    assert!(archive.starts_with(b"PK\x03\x04"));
    assert!(
        archive
            .windows(b"rank_u8.npy".len())
            .any(|window| window == b"rank_u8.npy")
    );
    assert!(
        archive
            .windows(b"delta_u8.npy".len())
            .any(|window| window == b"delta_u8.npy")
    );
    assert!(
        archive
            .windows(b"metadata.json".len())
            .any(|window| window == b"metadata.json")
    );
    std::fs::remove_file(&output).unwrap();
    std::fs::remove_dir(&folder).unwrap();
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
