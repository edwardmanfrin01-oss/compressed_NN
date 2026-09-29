use crate::compression::base_bits::{BaseBit, EncodingContext};
use crate::compression::base_selection::BaseSelectionContext;
use crate::compression::data::BitDataSet;
use crate::compression::entropy::ConstantBitPolarity;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;
use fxhash::FxHashMap;

#[derive(Clone)]
pub struct PreEncodeContext {
    pub bit_data: BitDataSet,
    pub row_to_base_id: Vec<usize>,
    pub layout: BaseLayoutInfo,
    pub variable_base_table: Vec<(BitVec<usize, Lsb0>, usize)>,
    /// Column indices into `variable_base_table` rows, ordered by ascending
    /// unweighted entropy as used by `BuildSortedBaseTable`.
    pub entropy_sorted_column_order: Option<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseLayoutInfo {
    pub bit_states: Vec<BaseBitLayoutState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseBitLayoutState {
    Deviation,
    Variable,
    ConstantZero,
    ConstantOne,
}

impl BaseLayoutInfo {
    pub fn from_bit_states(bit_states: Vec<BaseBitLayoutState>) -> Self {
        Self { bit_states }
    }

    pub fn chunk_size(&self) -> usize {
        self.bit_states.len()
    }

    pub fn state_at(&self, bit_position: usize) -> BaseBitLayoutState {
        self.bit_states
            .get(bit_position)
            .copied()
            .unwrap_or(BaseBitLayoutState::Deviation)
    }

    pub fn selected_base_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if matches!(
                    state,
                    BaseBitLayoutState::Variable
                        | BaseBitLayoutState::ConstantZero
                        | BaseBitLayoutState::ConstantOne
                ) {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn deviation_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::Deviation {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn variable_base_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::Variable {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn constant_zero_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::ConstantZero {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn constant_one_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::ConstantOne {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct BuildBaseTable {}

impl Filter for BuildBaseTable {
    type Input = BaseSelectionContext;
    type Output = PreEncodeContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building base table and layout from selected bases");
        let BaseSelectionContext {
            bit_data,
            base_bits,
            constant_bit_polarity,
        } = input;
        Ok(build_encode_context(
            bit_data,
            base_bits.as_ref(),
            &constant_bit_polarity,
            false,
        ))
    }
}

pub struct BuildSortedBaseTable {}

impl Filter for BuildSortedBaseTable {
    type Input = BaseSelectionContext;
    type Output = PreEncodeContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building and sorting base table");
        let BaseSelectionContext {
            bit_data,
            base_bits,
            constant_bit_polarity,
        } = input;
        Ok(build_encode_context(
            bit_data,
            base_bits.as_ref(),
            &constant_bit_polarity,
            true,
        ))
    }
}

fn build_encode_context(
    bit_data: BitDataSet,
    base_bit_groups: &dyn BaseBit,
    constant_bit_polarity: &ConstantBitPolarity,
    sort_bases: bool,
) -> PreEncodeContext {
    let _timer = ScopedTimer::info("build_encode_context");
    let EncodingContext {
        row_to_id: row_to_base_id,
        base_table: selected_base_table,
        sorted_column_order,
    } = base_bit_groups.get_encoding_context(&bit_data, sort_bases);
    drop(_timer);

    let base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
    let layout = build_base_layout_from_constant_polarity(
        bit_data.chunk_size(),
        &base_bit_positions,
        &constant_bit_polarity.constant_zero_bit_positions,
        &constant_bit_polarity.constant_one_bit_positions,
    );
    let variable_positions = layout.variable_base_bit_positions();
    let variable_base_table = project_selected_bases_to_variable(
        &base_bit_positions,
        &variable_positions,
        &selected_base_table,
    );

    let entropy_sorted_column_order = sorted_column_order.map(|selected_order| {
        map_selected_order_to_variable_order(
            &base_bit_positions,
            &variable_positions,
            &selected_order,
        )
    });

    PreEncodeContext {
        bit_data,
        row_to_base_id,
        layout,
        variable_base_table,
        entropy_sorted_column_order,
    }
}

fn map_selected_order_to_variable_order(
    selected_positions: &[usize],
    variable_positions: &[usize],
    selected_order: &[usize],
) -> Vec<usize> {
    let mut pos_to_var_idx =
        FxHashMap::with_capacity_and_hasher(variable_positions.len(), Default::default());
    for (var_idx, &pos) in variable_positions.iter().enumerate() {
        pos_to_var_idx.insert(pos, var_idx);
    }
    selected_order
        .iter()
        .filter_map(|&selected_idx| {
            let global_pos = selected_positions[selected_idx];
            pos_to_var_idx.get(&global_pos).copied()
        })
        .collect()
}

pub(crate) fn build_base_layout_from_constant_polarity(
    chunk_size: usize,
    selected_positions: &[usize],
    constant_zero_positions: &[usize],
    constant_one_positions: &[usize],
) -> BaseLayoutInfo {
    let mut bit_states = vec![BaseBitLayoutState::Deviation; chunk_size];

    for &bit_position in selected_positions {
        if bit_position >= bit_states.len() {
            continue;
        }
        bit_states[bit_position] = if constant_one_positions.contains(&bit_position) {
            BaseBitLayoutState::ConstantOne
        } else if constant_zero_positions.contains(&bit_position) {
            BaseBitLayoutState::ConstantZero
        } else {
            BaseBitLayoutState::Variable
        };
    }

    BaseLayoutInfo::from_bit_states(bit_states)
}

pub(crate) fn project_selected_bases_to_variable(
    selected_positions: &[usize],
    variable_positions: &[usize],
    selected_bases: &[(BitVec<usize, Lsb0>, usize)],
) -> Vec<(BitVec<usize, Lsb0>, usize)> {
    let variable_indices: Vec<usize> = variable_positions
        .iter()
        .filter_map(|bit_pos| selected_positions.iter().position(|p| p == bit_pos))
        .collect();

    selected_bases
        .iter()
        .map(|(selected_bits, count)| {
            let mut variable_bits = BitVec::with_capacity(variable_indices.len());
            for selected_idx in variable_indices.iter().copied() {
                variable_bits.push(
                    selected_bits
                        .get(selected_idx)
                        .map(|bit| *bit)
                        .unwrap_or(false),
                );
            }
            (variable_bits, *count)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_bits::BaseBitGroups;
    use crate::compression::data::{
        BitData, BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform,
    };
    use crate::compression::entropy::ConstantBitPolarity;

    fn create_test_bit_data_set(rows_and_bits: Vec<Vec<bool>>) -> BitDataSet {
        assert!(!rows_and_bits.is_empty(), "Must have at least one row");
        let chunk_size = rows_and_bits[0].len();
        let num_rows = rows_and_bits.len();

        let mut data = BitVec::with_capacity(num_rows * chunk_size);
        for row in &rows_and_bits {
            assert_eq!(row.len(), chunk_size, "All rows must have same bit width");
            for &bit in row {
                data.push(bit);
            }
        }

        let bit_data = BitData {
            data,
            chunk_size,
            stride: chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UInt(chunk_size as u16),
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to construct BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    #[test]
    fn test_build_base_table_populates_layout_and_rows() {
        let bit_data = create_test_bit_data_set(vec![
            vec![false, false, false, true],
            vec![false, true, false, true],
            vec![true, false, true, true],
            vec![true, true, true, true],
        ]);

        let mut groups = BaseBitGroups::new(4, 4);
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 1);
        groups.add_bit_position(&bit_data, 3);

        let context = BuildBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data,
                    Box::new(groups),
                    ConstantBitPolarity {
                        constant_zero_bit_positions: Vec::new(),
                        constant_one_bit_positions: vec![3],
                    },
                ),
            )
            .expect("BuildBaseTable should succeed");

        assert_eq!(context.variable_base_table.len(), 4);
        assert_eq!(context.layout.selected_base_bit_positions(), vec![0, 1, 3]);
        assert_eq!(context.layout.variable_base_bit_positions(), vec![0, 1]);
        assert_eq!(context.layout.constant_one_bit_positions(), vec![3]);
        assert!(context.layout.constant_zero_bit_positions().is_empty());
        assert!(context.entropy_sorted_column_order.is_none());
        let mut ids = context.row_to_base_id.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_build_sorted_base_table_remaps_ids_to_sorted_order() {
        let bit_data = create_test_bit_data_set(vec![
            vec![true, true, false, false],
            vec![false, false, false, false],
            vec![true, false, false, false],
            vec![false, true, false, false],
        ]);

        let mut groups = BaseBitGroups::new(4, 4);
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 1);

        let unsorted = BuildBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data.clone(),
                    Box::new(groups.clone()),
                    ConstantBitPolarity::default(),
                ),
            )
            .expect("BuildBaseTable should succeed");
        let sorted = BuildSortedBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data,
                    Box::new(groups),
                    ConstantBitPolarity::default(),
                ),
            )
            .expect("BuildSortedBaseTable should succeed");

        assert_eq!(
            sorted.variable_base_table.len(),
            unsorted.variable_base_table.len()
        );
        assert!(unsorted.entropy_sorted_column_order.is_none());
        assert!(sorted.entropy_sorted_column_order.is_some());

        let mut unsorted_patterns = unsorted
            .variable_base_table
            .iter()
            .map(|(bits, _)| bits.clone())
            .collect::<Vec<_>>();
        let mut sorted_patterns = sorted
            .variable_base_table
            .iter()
            .map(|(bits, _)| bits.clone())
            .collect::<Vec<_>>();
        unsorted_patterns.sort();
        sorted_patterns.sort();
        assert_eq!(sorted_patterns, unsorted_patterns);
    }

    #[test]
    fn test_sorted_base_table_uses_column_entropy_priority_lexicographic_order() {
        let bit_data = create_test_bit_data_set(vec![
            vec![true, false, true, false],  // row 0: 1010
            vec![false, false, false, true], // row 1: 0001
            vec![true, false, true, true],   // row 2: 1011
            vec![false, true, true, true],   // row 3: 0111
        ]);

        let mut groups = BaseBitGroups::new(4, 4);
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 1);
        groups.add_bit_position(&bit_data, 2);
        groups.add_bit_position(&bit_data, 3);

        let sorted = BuildSortedBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data,
                    Box::new(groups),
                    ConstantBitPolarity::default(),
                ),
            )
            .expect("BuildSortedBaseTable should succeed");

        let as_vec = |bits: &BitVec<usize, Lsb0>| -> Vec<bool> { bits.iter().by_vals().collect() };
        let sorted_rows: Vec<Vec<bool>> = sorted
            .variable_base_table
            .iter()
            .map(|(bits, _)| as_vec(bits))
            .collect();

        assert_eq!(
            sorted_rows,
            vec![
                vec![false, true, true, true],
                vec![true, false, true, true],
                vec![true, false, true, false],
                vec![false, false, false, true],
            ]
        );
        assert_eq!(sorted.row_to_base_id, vec![2, 3, 1, 0]);
        assert_eq!(sorted.entropy_sorted_column_order, Some(vec![1, 2, 3, 0]));
    }
}
