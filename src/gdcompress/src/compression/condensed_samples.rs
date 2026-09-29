use crate::compression::base_bits::BaseBitGroups;
use crate::compression::data::BitDataSet;
use crate::compression::encoding::CondensedSamples;
use crate::compression::entropy::{ConstantBitPolarity, EntropyScoredContext};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

fn organize_entropy_by_feature(
    entropy: &[(usize, f64)],
    bit_data: &BitDataSet,
) -> Vec<(usize, f64)> {
    let num_features = bit_data.num_features();
    let mut entropy_by_feature: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_features];
    let mut zero_entropy = Vec::new();

    for &(bit_pos, entropy_val) in entropy {
        if let Some(feature_idx) = bit_data.feature_index_for_bit(bit_pos) {
            if entropy_val == 0.0 {
                zero_entropy.push((bit_pos, entropy_val));
            } else {
                entropy_by_feature[feature_idx].push((bit_pos, entropy_val));
            }
        }
    }

    for feature_entropy in &mut entropy_by_feature {
        feature_entropy.sort_by(|a, b| b.0.cmp(&a.0));
    }

    let mut result = zero_entropy;
    let mut pointers = vec![0usize; num_features];
    loop {
        let mut candidates: Vec<((usize, f64), usize)> = Vec::with_capacity(num_features);
        for (feat_idx, feature_entropy) in entropy_by_feature.iter().enumerate() {
            if let Some(&entry) = feature_entropy.get(pointers[feat_idx]) {
                candidates.push((entry, feat_idx));
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|a, b| {
            b.0 .1
                .partial_cmp(&a.0 .1)
                .unwrap()
                .then_with(|| b.0 .0.cmp(&a.0 .0))
        });
        let ((bit_pos, entropy), chosen_feat_idx) = candidates[0];
        result.push((bit_pos, entropy));
        pointers[chosen_feat_idx] += 1;
    }

    result
}

pub struct GenCondensedSamples {
    pub m_max: usize,
}

impl Filter for GenCondensedSamples {
    type Input = EntropyScoredContext;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Generating condensed samples (m_max = {})",
            self.m_max
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        } = input;
        let original_num_rows = bit_data.num_rows();
        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let condensed_samples =
            select_condensed_samples(&bit_data, &entropy_scores, self.m_max, &mut bb);

        let bit_data = append_condensed_samples(bit_data, condensed_samples);
        let constant_bit_polarity = update_constant_bit_polarity_from_appended_rows(
            &bit_data,
            &constant_bit_polarity,
            original_num_rows,
        );
        Ok(EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        })
    }
}

fn update_constant_bit_polarity_from_appended_rows(
    bit_data: &BitDataSet,
    previous: &ConstantBitPolarity,
    appended_row_start: usize,
) -> ConstantBitPolarity {
    fn appended_rows_match(
        bit_data: &BitDataSet,
        bit_position: usize,
        expected: bool,
        appended_row_start: usize,
    ) -> bool {
        if bit_position >= bit_data.chunk_size() {
            return false;
        }

        let start = appended_row_start.min(bit_data.num_rows());
        for row in start..bit_data.num_rows() {
            if unsafe { bit_data.get_bit_unchecked(row, bit_position) } != expected {
                return false;
            }
        }

        true
    }

    let constant_zero_bit_positions = previous
        .constant_zero_bit_positions
        .iter()
        .copied()
        .filter(|&bit_position| {
            appended_rows_match(bit_data, bit_position, false, appended_row_start)
        })
        .collect();
    let constant_one_bit_positions = previous
        .constant_one_bit_positions
        .iter()
        .copied()
        .filter(|&bit_position| {
            appended_rows_match(bit_data, bit_position, true, appended_row_start)
        })
        .collect();

    ConstantBitPolarity {
        constant_zero_bit_positions,
        constant_one_bit_positions,
    }
}

fn select_condensed_samples(
    bit_data: &BitDataSet,
    entropy: &[(usize, f64)],
    m_max: usize,
    base_bits: &mut BaseBitGroups,
) -> CondensedSamples {
    fn bits_to_u64(bits: &BitSlice<usize, Lsb0>) -> u64 {
        if bits.is_empty() {
            0
        } else {
            bits.load_le::<u64>()
        }
    }

    fn push_u64_bits(stream: &mut BitVec<usize, Lsb0>, value: u64, bits: usize) {
        if bits == 0 {
            return;
        }

        let start = stream.len();
        stream.resize(start + bits, false);
        stream[start..].store_le(value);
    }

    fn mask_for_bits(bits: usize) -> u64 {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    let mut samples = Vec::new();
    let mut weights = Vec::new();

    let organized_entropy_by_feature = organize_entropy_by_feature(entropy, bit_data);

    let zero_entropy_bits: Vec<usize> = organized_entropy_by_feature
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val == 0.0)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    base_bits.add_constant_bit_positions(&zero_entropy_bits);

    for &(bit_position, _entropy_val) in organized_entropy_by_feature
        .iter()
        .skip(zero_entropy_bits.len())
    {
        if base_bits.get_num_bases() >= m_max {
            break;
        }
        base_bits.add_bit_position(bit_data, bit_position);
        tracing::debug!(
            bit_position,
            current_num_bases = base_bits.get_num_bases(),
            m_max,
            "added bit position to condensed samples"
        );
    }

    let base_groups = base_bits.get_groups();
    let base_mask = base_bits.get_base_bit_mask().to_bitvec();

    for base_group in base_groups.iter() {
        let padded_base =
            BitVec::from_bitslice(unsafe { bit_data.get_chunk_unchecked(base_group[0]) })
                & base_mask.clone();

        let mut condensed_bitvec = BitVec::<usize, Lsb0>::with_capacity(bit_data.chunk_size());

        for feature_idx in 0..bit_data.num_features() {
            let feature_offset = bit_data.feature_offset(feature_idx);
            let feature_bits = bit_data.feature_bits(feature_idx);
            let feature_end = feature_offset + feature_bits;

            let base_feature = &padded_base[feature_offset..feature_end];
            let base_feature_mask = &base_mask[feature_offset..feature_end];

            let base_value = bits_to_u64(base_feature);
            let deviation_mask = !bits_to_u64(base_feature_mask) & mask_for_bits(feature_bits);

            let mut deviation_accumulator = 0u128;
            for &sample_idx in base_group.iter() {
                let chunk = unsafe { bit_data.get_chunk_unchecked(sample_idx) };
                let sample_feature = unsafe { chunk.get_unchecked(feature_offset..feature_end) };
                let sample_feature_value = bits_to_u64(sample_feature);
                let feature_deviation = sample_feature_value & deviation_mask;
                deviation_accumulator += feature_deviation as u128;
            }

            let mean_deviation = (deviation_accumulator / base_group.len() as u128) as u64;
            let condensed_feature_value =
                (base_value + mean_deviation) & mask_for_bits(feature_bits);
            push_u64_bits(&mut condensed_bitvec, condensed_feature_value, feature_bits);
        }

        samples.push(condensed_bitvec);
        weights.push(base_group.len());
    }

    CondensedSamples { samples, weights }
}

fn append_condensed_samples(
    mut bit_data: BitDataSet,
    condensed_samples: CondensedSamples,
) -> BitDataSet {
    bit_data
        .info
        .set_condensed_sample_weights(Some(condensed_samples.weights));

    for sample in condensed_samples.samples {
        bit_data
            .data
            .extend_from_bitslice(sample.as_bitslice())
            .expect("condensed sample chunk size must match dataset chunk size");
    }
    bit_data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::data::{BitDataInfo, FeatureDataType, FeatureSpec};

    // ============================================================================
    // Helper Functions for Test Data Construction
    // ============================================================================

    /// Helper struct to build test BitDataSet with flexible configuration
    struct TestBitDataBuilder {
        num_rows: usize,
        num_features: usize,
        bits_per_feature: usize,
    }

    impl TestBitDataBuilder {
        fn new(num_rows: usize, num_features: usize, bits_per_feature: usize) -> Self {
            Self {
                num_rows,
                num_features,
                bits_per_feature,
            }
        }

        /// Build a BitDataSet with default data (all zeros except specified bits)
        fn build(self) -> BitDataSet {
            let chunk_size = self.num_features * self.bits_per_feature;
            let total_bits = chunk_size * self.num_rows;
            let mut data = BitVec::<usize, Lsb0>::with_capacity(total_bits);

            for _ in 0..self.num_rows {
                for _ in 0..chunk_size {
                    data.push(false);
                }
            }

            let features = (0..self.num_features)
                .map(|_| FeatureSpec {
                    data_type: FeatureDataType::UInt(self.bits_per_feature as u16),
                    transform: crate::compression::data::FeatureTransform::None,
                })
                .collect();

            let info =
                BitDataInfo::new(features, total_bits).expect("failed to create BitDataInfo");

            let data_struct = crate::compression::data::BitData {
                data,
                chunk_size,
                stride: chunk_size,
                num_rows: self.num_rows,
            };

            BitDataSet {
                data: data_struct,
                info,
            }
        }
    }

    /// Create a simple test entropy vector
    fn create_test_entropy(num_bits: usize) -> Vec<(usize, f64)> {
        (0..num_bits).map(|i| (i, (i as f64) * 0.1)).collect()
    }

    /// Create a test entropy with some zero entropy bits
    fn create_entropy_with_zeros(num_bits: usize, zero_count: usize) -> Vec<(usize, f64)> {
        let mut entropy = vec![];
        // Add zero entropy bits
        for i in 0..zero_count {
            entropy.push((i, 0.0));
        }
        // Add non-zero entropy bits
        for i in zero_count..num_bits {
            entropy.push((i, (i as f64) * 0.1));
        }
        entropy
    }

    // ============================================================================
    // Tests for organize_entropy_by_feature
    // ============================================================================

    #[test]
    fn test_organize_entropy_by_feature_empty_entropy() {
        let bit_data = TestBitDataBuilder::new(10, 2, 4).build();
        let entropy = vec![];

        let result = organize_entropy_by_feature(&entropy, &bit_data);

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_organize_entropy_by_feature_single_feature() {
        let bit_data = TestBitDataBuilder::new(10, 1, 4).build();
        let entropy = create_test_entropy(4);

        let result = organize_entropy_by_feature(&entropy, &bit_data);

        assert_eq!(result.len(), entropy.len());
    }

    #[test]
    fn test_organize_entropy_by_feature_zero_entropy_first() {
        let bit_data = TestBitDataBuilder::new(10, 2, 4).build();
        let entropy = create_entropy_with_zeros(8, 2);

        let result = organize_entropy_by_feature(&entropy, &bit_data);

        // First two should be zero entropy
        assert_eq!(result[0].1, 0.0);
        assert_eq!(result[1].1, 0.0);

        // Rest should be organized by feature in ascending entropy order within each feature
        for entry in result.iter().skip(2) {
            assert!(entry.1 > 0.0);
        }
    }

    #[test]
    fn test_organize_entropy_by_feature_multiple_features() {
        let bit_data = TestBitDataBuilder::new(10, 3, 3).build();
        let entropy = create_test_entropy(9);

        let result = organize_entropy_by_feature(&entropy, &bit_data);

        assert_eq!(result.len(), 9);
        // Should have been reorganized but all entropy values should be present
        let result_entropies: Vec<f64> = result.iter().map(|(_, e)| *e).collect();
        let expected_entropies: Vec<f64> = entropy.iter().map(|(_, e)| *e).collect();
        assert_eq!(result_entropies.len(), expected_entropies.len());
    }

    // ============================================================================
    // Tests for select_condensed_samples
    // ============================================================================

    #[test]
    fn test_select_condensed_samples_simple() {
        let bit_data = TestBitDataBuilder::new(4, 1, 8).build();
        let entropy = create_test_entropy(8);

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 2, &mut bb);

        assert_eq!(result.samples.len(), result.weights.len());
        for sample in &result.samples {
            assert_eq!(sample.len(), bit_data.chunk_size());
        }
    }

    #[test]
    fn test_select_condensed_samples_m_max_zero() {
        let bit_data = TestBitDataBuilder::new(4, 1, 8).build();
        let entropy = create_test_entropy(8);

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 0, &mut bb);

        // With m_max = 0, we should only have empty groups or samples with zero-entropy bits
        assert!(result.samples.is_empty() || result.weights.iter().sum::<usize>() > 0);
    }

    #[test]
    fn test_select_condensed_samples_respects_m_max() {
        let bit_data = TestBitDataBuilder::new(10, 1, 16).build();
        let entropy = create_test_entropy(16);

        let m_max = 4;
        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, m_max, &mut bb);

        // Number of groups/bases should not exceed m_max
        assert!(result.samples.len() <= m_max);
    }

    #[test]
    fn test_select_condensed_samples_weights_sum() {
        let bit_data = TestBitDataBuilder::new(10, 1, 8).build();
        let entropy = create_test_entropy(8);

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 3, &mut bb);

        // All rows should be accounted for in weights
        let total_weight: usize = result.weights.iter().sum();
        assert_eq!(total_weight, bit_data.num_rows());
    }

    #[test]
    fn test_select_condensed_samples_empty_entropy() {
        let bit_data = TestBitDataBuilder::new(4, 1, 8).build();
        let entropy = vec![];

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 2, &mut bb);

        assert_eq!(result.samples.len(), 1); // Single group with all rows
        assert_eq!(result.weights, vec![4]); // All 4 rows in one group
    }

    #[test]
    fn test_select_condensed_samples_with_zero_entropy_bits() {
        let bit_data = TestBitDataBuilder::new(6, 1, 8).build();
        let entropy = create_entropy_with_zeros(8, 3);

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 2, &mut bb);

        assert_eq!(result.samples.len(), result.weights.len());
        let total_weight: usize = result.weights.iter().sum();
        assert_eq!(total_weight, 6);
    }

    // ============================================================================
    // Tests for append_condensed_samples
    // ============================================================================

    #[test]
    fn test_append_condensed_samples_empty() {
        let bit_data = TestBitDataBuilder::new(5, 1, 8).build();
        let condensed_samples = CondensedSamples {
            samples: vec![],
            weights: vec![],
        };

        let original_num_rows = bit_data.num_rows();
        let result = append_condensed_samples(bit_data, condensed_samples);

        assert_eq!(result.num_rows(), original_num_rows);
    }

    #[test]
    fn test_append_condensed_samples_single_sample() {
        let bit_data = TestBitDataBuilder::new(3, 1, 8).build();
        let mut sample = BitVec::<usize, Lsb0>::with_capacity(8);
        for _ in 0..8 {
            sample.push(true);
        }
        let condensed_samples = CondensedSamples {
            samples: vec![sample],
            weights: vec![2],
        };

        let original_num_rows = bit_data.num_rows();
        let result = append_condensed_samples(bit_data, condensed_samples);

        assert_eq!(result.num_rows(), original_num_rows + 1);
    }

    #[test]
    fn test_append_condensed_samples_multiple_samples() {
        let bit_data = TestBitDataBuilder::new(4, 1, 8).build();
        let mut samples = vec![];
        for _ in 0..3 {
            let mut sample = BitVec::<usize, Lsb0>::with_capacity(8);
            for _ in 0..8 {
                sample.push(false);
            }
            samples.push(sample);
        }
        let condensed_samples = CondensedSamples {
            samples,
            weights: vec![2, 1, 1],
        };

        let original_num_rows = bit_data.num_rows();
        let result = append_condensed_samples(bit_data, condensed_samples);

        assert_eq!(result.num_rows(), original_num_rows + 3);
    }

    #[test]
    fn test_append_condensed_samples_stores_weights() {
        let bit_data = TestBitDataBuilder::new(2, 1, 8).build();
        let weights = vec![3, 2];
        let samples = weights
            .iter()
            .map(|_| {
                let mut s = BitVec::<usize, Lsb0>::with_capacity(8);
                for _ in 0..8 {
                    s.push(false);
                }
                s
            })
            .collect();
        let condensed_samples = CondensedSamples { samples, weights };

        let result = append_condensed_samples(bit_data, condensed_samples);

        assert_eq!(
            result.info.m_condensed_sample_weights(),
            Some(vec![3, 2].as_slice())
        );
    }

    // ============================================================================
    // Tests for GenCondensedSamples Filter
    // ============================================================================

    #[test]
    fn test_gen_condensed_samples_filter_process() {
        let bit_data = TestBitDataBuilder::new(8, 1, 8).build();
        let entropy = create_test_entropy(8);
        let m_max = 3;

        let filter = GenCondensedSamples { m_max };
        let result = filter.process(EntropyScoredContext::new(bit_data.clone(), entropy.clone()));

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.entropy_scores, entropy); // Entropy should pass through unchanged
        assert!(output.bit_data.num_rows() >= bit_data.num_rows()); // May have condensed samples added
    }

    #[test]
    fn test_gen_condensed_samples_filter_m_max_respected() {
        let bit_data = TestBitDataBuilder::new(10, 1, 16).build();
        let entropy = create_test_entropy(16);
        let m_max = 5;

        let filter = GenCondensedSamples { m_max };
        let output = filter
            .process(EntropyScoredContext::new(bit_data, entropy))
            .expect("filter should succeed");

        if let Some(weights) = output.bit_data.info.m_condensed_sample_weights() {
            // Number of condensed samples should respect m_max
            assert!(weights.len() <= m_max);
        }
    }

    #[test]
    fn test_gen_condensed_samples_filter_entropy_preserved() {
        let bit_data = TestBitDataBuilder::new(5, 1, 8).build();
        let entropy = create_test_entropy(8);

        let filter = GenCondensedSamples { m_max: 2 };
        let output = filter
            .process(EntropyScoredContext::new(bit_data, entropy.clone()))
            .expect("filter should succeed");

        assert_eq!(output.entropy_scores, entropy);
    }

    #[test]
    fn test_gen_condensed_samples_filter_recomputes_constant_bit_polarity() {
        let bit_data = TestBitDataBuilder::new(4, 1, 4).build();
        let entropy = create_test_entropy(4);
        let input = EntropyScoredContext {
            bit_data,
            entropy_scores: entropy,
            constant_bit_polarity: ConstantBitPolarity {
                constant_zero_bit_positions: vec![0, 1],
                constant_one_bit_positions: vec![2, 3],
            },
        };

        let filter = GenCondensedSamples { m_max: 2 };
        let output = filter.process(input).expect("filter should succeed");

        assert_eq!(
            output.constant_bit_polarity.constant_zero_bit_positions,
            vec![0, 1]
        );
        assert!(
            output
                .constant_bit_polarity
                .constant_one_bit_positions
                .is_empty()
        );
    }

    #[test]
    fn test_update_constant_polarity_checks_only_appended_rows() {
        let mut bit_data = TestBitDataBuilder::new(3, 1, 2).build();
        // Row 0: [1, 1] (breaks prior expectations)
        // Row 1: [0, 0]
        // Row 2: [0, 1] (appended row)
        bit_data.data.data.set(0, true);
        bit_data.data.data.set(1, true);
        bit_data.data.data.set(4, false);
        bit_data.data.data.set(5, true);

        let previous = ConstantBitPolarity {
            constant_zero_bit_positions: vec![0],
            constant_one_bit_positions: vec![1],
        };

        let updated = update_constant_bit_polarity_from_appended_rows(&bit_data, &previous, 2);

        assert_eq!(updated.constant_zero_bit_positions, vec![0]);
        assert_eq!(updated.constant_one_bit_positions, vec![1]);
    }

    #[test]
    fn test_update_constant_polarity_removes_on_appended_row_mismatch() {
        let mut bit_data = TestBitDataBuilder::new(3, 1, 2).build();
        // Row 2 (appended): [1, 1], so bit 0 mismatches expected false.
        bit_data.data.data.set(4, true);
        bit_data.data.data.set(5, true);

        let previous = ConstantBitPolarity {
            constant_zero_bit_positions: vec![0],
            constant_one_bit_positions: vec![1],
        };

        let updated = update_constant_bit_polarity_from_appended_rows(&bit_data, &previous, 2);

        assert!(updated.constant_zero_bit_positions.is_empty());
        assert_eq!(updated.constant_one_bit_positions, vec![1]);
    }

    #[test]
    fn test_gen_condensed_samples_filter_zero_m_max() {
        let bit_data = TestBitDataBuilder::new(4, 1, 8).build();
        let entropy = create_test_entropy(8);

        let filter = GenCondensedSamples { m_max: 0 };
        let result = filter.process(EntropyScoredContext::new(bit_data, entropy));

        assert!(result.is_ok());
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    #[test]
    fn test_condensed_samples_end_to_end() {
        let bit_data = TestBitDataBuilder::new(10, 2, 8).build();
        let entropy = create_entropy_with_zeros(16, 2);

        let filter = GenCondensedSamples { m_max: 4 };
        let output = filter
            .process(EntropyScoredContext::new(bit_data.clone(), entropy))
            .expect("filter should succeed");
        let output_bit_data = output.bit_data;

        // Verify basic properties
        assert!(output_bit_data.num_rows() >= bit_data.num_rows());
        assert_eq!(output_bit_data.chunk_size(), bit_data.chunk_size());

        // Verify condensed samples were created
        if output_bit_data.num_rows() > bit_data.num_rows() {
            assert!(output_bit_data.info.m_condensed_sample_weights().is_some());
        }
    }

    #[test]
    fn test_condensed_samples_preserves_chunk_size() {
        let chunk_size = 12;
        let bit_data = TestBitDataBuilder::new(8, 3, 4).build();
        assert_eq!(bit_data.chunk_size(), chunk_size);

        let entropy = create_test_entropy(chunk_size);
        let filter = GenCondensedSamples { m_max: 3 };
        let output = filter
            .process(EntropyScoredContext::new(bit_data, entropy))
            .expect("filter should succeed");
        let output_bit_data = output.bit_data;

        assert_eq!(output_bit_data.chunk_size(), chunk_size);
    }

    #[test]
    fn test_select_condensed_samples_with_multiple_features() {
        let bit_data = TestBitDataBuilder::new(8, 4, 4).build();
        let entropy = create_entropy_with_zeros(16, 1);

        let mut bb = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let result = select_condensed_samples(&bit_data, &entropy, 4, &mut bb);

        // Verify samples have correct size
        for sample in &result.samples {
            assert_eq!(sample.len(), 16);
        }

        // Verify weights and samples match
        assert_eq!(result.samples.len(), result.weights.len());
    }
}
