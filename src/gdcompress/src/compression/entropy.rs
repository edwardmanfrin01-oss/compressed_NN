use crate::compression::data::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

pub type EntropyBitScore = (usize, f64);

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConstantBitPolarity {
    pub constant_zero_bit_positions: Vec<usize>,
    pub constant_one_bit_positions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct EntropyScoredContext {
    pub bit_data: BitDataSet,
    pub entropy_scores: Vec<EntropyBitScore>,
    pub constant_bit_polarity: ConstantBitPolarity,
}

impl EntropyScoredContext {
    pub fn new(bit_data: BitDataSet, entropy_scores: Vec<EntropyBitScore>) -> Self {
        let constant_bit_polarity =
            derive_constant_bit_polarity_from_entropy(&bit_data, &entropy_scores);
        Self {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        }
    }
}

fn derive_constant_bit_polarity_from_entropy(
    bit_data: &BitDataSet,
    entropy_scores: &[EntropyBitScore],
) -> ConstantBitPolarity {
    let mut constant_zero_bit_positions = Vec::new();
    let mut constant_one_bit_positions = Vec::new();

    if bit_data.num_rows() == 0 {
        for &(bit_position, entropy_val) in entropy_scores {
            if entropy_val == 0.0 {
                constant_zero_bit_positions.push(bit_position);
            }
        }

        return ConstantBitPolarity {
            constant_zero_bit_positions,
            constant_one_bit_positions,
        };
    }

    for &(bit_position, entropy_val) in entropy_scores {
        if entropy_val != 0.0 {
            continue;
        }

        if unsafe { bit_data.get_bit_unchecked(0, bit_position) } {
            constant_one_bit_positions.push(bit_position);
        } else {
            constant_zero_bit_positions.push(bit_position);
        }
    }

    ConstantBitPolarity {
        constant_zero_bit_positions,
        constant_one_bit_positions,
    }
}

pub struct Entropy;

impl Filter for Entropy {
    type Input = BitDataSet;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Calculating entropy for each bit position");
        let entropy = calculate_entropy(&input);
        Ok(EntropyScoredContext::new(input, entropy))
    }
}

pub fn calculate_entropy(bit_data: &BitDataSet) -> Vec<EntropyBitScore> {
    let num_rows = bit_data.num_rows();
    let chunk_size = bit_data.chunk_size();
    if num_rows == 0 {
        return (0..chunk_size).map(|bit| (bit, 0.0)).collect();
    }

    let inv_rows = 1.0 / num_rows as f64;
    let mut out = Vec::with_capacity(chunk_size);

    for bit in 0..chunk_size {
        let mut count_ones = 0usize;
        for row in 0..num_rows {
            if unsafe { bit_data.get_bit_unchecked(row, bit) } {
                count_ones += 1;
            }
        }

        let entropy = if count_ones == 0 || count_ones == num_rows {
            0.0
        } else {
            let p = count_ones as f64 * inv_rows;
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        };
        out.push((bit, entropy));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::data::{
        BitData, BitDataInfo, BitDataSet, FeatureDataType, FeatureSpec, FeatureTransform,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn test_entropy_calculation_1() {
        // Create a simple BitData for testing
        let num_rows = 4;
        let chunk_size = 3;
        let num_features = 3;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, // Row 0
            true, true, false, // Row 1
            false, false, true, // Row 2
            false, true, false, // Row 3
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(bits_per_feature as u16),
                transform: FeatureTransform::None
            };
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        tracing::info!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        // Add more assertions based on expected entropy values
    }

    #[test]
    fn test_entropy_calculation_2() {
        // Create a simple BitData for testing
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(bits_per_feature as u16),
                transform: FeatureTransform::None
            };
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        let expected_entropies = vec![
            (0, 0.9183),
            (1, 0.9183),
            (2, 0.9183),
            (3, 0.9183),
            (4, 0.65002),
            (5, 0.65002),
            (6, 0.9183),
            (7, 1.0),
        ];
        tracing::info!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        assert_eq!(
            entropies
                .iter()
                .map(|(i, e)| (*i, (e * 100_000.0).round() / 100_000.0))
                .collect::<Vec<_>>(),
            expected_entropies
        );
    }

    #[test]
    fn test_pipeline_with_entropy() {
        // Create a simple BitData for testing
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(bits_per_feature as u16),
                transform: FeatureTransform::None
            };
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropy_filter = Entropy;
        let entropy_scored = entropy_filter.process(bit_data).unwrap();
        tracing::info!(
            "Entropies from pipeline: {:?}",
            entropy_scored.entropy_scores
        );
        assert_eq!(entropy_scored.entropy_scores.len(), chunk_size);
    }

    #[test]
    fn test_entropy_context_derives_constant_bit_polarity() {
        let num_rows = 4;
        let chunk_size = 4;
        let data = vec![
            false, true, true, false, // Row 0
            false, true, false, false, // Row 1
            false, true, true, true, // Row 2
            false, true, false, true, // Row 3
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UInt(1),
                transform: FeatureTransform::None
            };
            chunk_size
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let context = EntropyScoredContext::new(bit_data, vec![(0, 0.0), (1, 0.0), (2, 1.0)]);

        assert_eq!(
            context.constant_bit_polarity.constant_zero_bit_positions,
            vec![0]
        );
        assert_eq!(
            context.constant_bit_polarity.constant_one_bit_positions,
            vec![1]
        );
    }
}
