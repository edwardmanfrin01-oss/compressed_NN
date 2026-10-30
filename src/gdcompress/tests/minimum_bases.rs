use gdcompress::compression::base_bits::{BaseBit, BaseBitGroups, BaseBitHyperLogLogCount};
use gdcompress::compression::base_selection::EnforceMinimumBases;
use gdcompress::prelude::*;
use gdcompress::BaseSelectionContext;
use image::{DynamicImage, Rgb, RgbImage};

fn fixture(approximate: bool, uniform: bool) -> BaseSelectionContext {
    let pixels = RgbImage::from_fn(4, 4, |x, y| {
        Rgb([if uniform { 0 } else { (4 * y + x) as u8 }, 0, 0])
    });
    let data = BuildImageBitDataSet {
        color_model: ImageColorModel::Rgb,
        pixel_grouping: PixelGrouping::new(1, 1),
        grouping_transform: ImageGroupingTransform::Raw,
        ..Default::default()
    }.process(DynamicImage::ImageRgb8(pixels)).unwrap();
    let scored = Entropy.process(data).unwrap();
    let constants: Vec<_> = scored.entropy_scores.iter()
        .filter(|(_, h)| *h == 0.0).map(|(p, _)| *p).collect();
    let mut bits: Box<dyn BaseBit> = if approximate {
        Box::new(BaseBitHyperLogLogCount::new(scored.bit_data.num_rows(), scored.bit_data.chunk_size()))
    } else {
        Box::new(BaseBitGroups::new(scored.bit_data.num_rows(), scored.bit_data.chunk_size()))
    };
    bits.add_constant_bit_positions(&constants);
    if let Some((position, _)) = scored.entropy_scores.iter().find(|(_, h)| *h > 0.0) {
        bits.add_bit_position(&scored.bit_data, *position);
    }
    BaseSelectionContext::new(scored.bit_data, bits, scored.constant_bit_polarity)
}

#[test]
fn disabled_and_already_sufficient_preserve_selection() {
    for minimum in [None, Some(1), Some(2)] {
        let input = fixture(false, false);
        let positions = input.base_bits.get_base_bit_positions().to_vec();
        let output = EnforceMinimumBases { k_min: minimum }.process(input).unwrap();
        assert_eq!(positions, output.base_bits.get_base_bit_positions());
    }
}

#[test]
fn exact_extension_stops_at_first_sufficient_partition_including_overshoot() {
    for approximate in [false, true] {
        let output = EnforceMinimumBases { k_min: Some(5) }
            .process(fixture(approximate, false)).unwrap();
        // Four independent variable bits: 2 -> 4 -> 8. Stop before 16.
        assert_eq!(output.base_bits.get_num_bases(), 8);
        let context = output.base_bits.get_encoding_context(&output.bit_data, false);
        assert_eq!(context.base_table.len(), 8);
    }
}

#[test]
fn unreachable_minimum_keeps_all_available_distinction() {
    for uniform in [false, true] {
        let output = EnforceMinimumBases { k_min: Some(32) }
            .process(fixture(false, uniform)).unwrap();
        assert_eq!(output.base_bits.get_num_bases(), if uniform { 1 } else { 16 });
        assert_eq!(output.base_bits.get_num_bits_per_base(), output.bit_data.chunk_size());
    }
}

#[test]
fn zero_minimum_is_rejected() {
    assert!(EnforceMinimumBases { k_min: Some(0) }
        .process(fixture(false, false)).is_err());
}
