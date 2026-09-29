use crate::compression::data::BitDataSet;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;
use fxhash::{FxHashMap, FxHashSet};
use rayon::prelude::*;
use std::cmp::Ordering;

pub struct EncodingContext {
    pub row_to_id: Vec<usize>,
    pub base_table: Vec<(BitVec<usize, Lsb0>, usize)>,
    pub sorted_column_order: Option<Vec<usize>>,
}

pub trait BaseBit {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize;
    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let mut num_bases = self.get_num_bases();
        for &bit_position in bit_positions {
            num_bases = self.add_bit_position(bit_data, bit_position);
        }
        num_bases
    }
    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize;
    fn get_encoding_context(&self, bit_data: &BitDataSet, sort: bool) -> EncodingContext;
    fn get_num_bases(&self) -> usize;
    fn get_num_bits_per_base(&self) -> usize;
    fn get_base_bit_mask(&self) -> &BitSlice<usize, Lsb0>;
    fn get_base_bit_positions(&self) -> &[usize];
}

fn initial_groups(num_rows: usize, step: usize) -> Vec<Vec<usize>> {
    vec![(0..num_rows).step_by(step).collect()]
}

fn collect_new_bit_positions(
    base_bit_mask: &BitSlice<usize, Lsb0>,
    bit_positions: &[usize],
) -> Vec<usize> {
    let mut seen = FxHashSet::with_capacity_and_hasher(bit_positions.len(), Default::default());
    let mut new_bit_positions = Vec::with_capacity(bit_positions.len());

    for &bit_position in bit_positions {
        assert!(
            bit_position < base_bit_mask.len(),
            "bit position {} out of bounds for chunk size {}",
            bit_position,
            base_bit_mask.len()
        );
        if base_bit_mask[bit_position] || !seen.insert(bit_position) {
            continue;
        }
        new_bit_positions.push(bit_position);
    }

    new_bit_positions
}

fn apply_selected_bit_positions(
    base_bit_mask: &mut BitVec<usize, Lsb0>,
    base_bit_positions: &mut Vec<usize>,
    bit_positions: &[usize],
) -> usize {
    let mut seen = FxHashSet::with_capacity_and_hasher(bit_positions.len(), Default::default());
    let mut added_count = 0usize;

    for &bit_position in bit_positions {
        assert!(
            bit_position < base_bit_mask.len(),
            "bit position {} out of bounds for chunk size {}",
            bit_position,
            base_bit_mask.len()
        );
        if base_bit_mask[bit_position] || !seen.insert(bit_position) {
            continue;
        }
        base_bit_mask.set(bit_position, true);
        base_bit_positions.push(bit_position);
        added_count += 1;
    }

    added_count
}

fn add_constant_bits(
    base_bit_mask: &mut BitVec<usize, Lsb0>,
    base_bit_positions: &mut Vec<usize>,
    num_bits_per_base: &mut usize,
    bit_positions: &[usize],
) -> usize {
    let added_count =
        apply_selected_bit_positions(base_bit_mask, base_bit_positions, bit_positions);
    *num_bits_per_base += added_count;
    added_count
}

fn get_selected_bases_from_groups(
    groups: &[Vec<usize>],
    base_bit_positions: &[usize],
    num_bases: usize,
    bit_data: &BitDataSet,
) -> Vec<(BitVec<usize, Lsb0>, usize)> {
    let mut bases = Vec::with_capacity(num_bases);
    for group in groups {
        if group.is_empty() {
            continue;
        }
        let chunk = unsafe { bit_data.get_chunk_unchecked(group[0]) };
        let mut packed_base = BitVec::with_capacity(base_bit_positions.len());
        for &bit_pos in base_bit_positions {
            packed_base.push(unsafe { *chunk.get_unchecked(bit_pos) });
        }
        bases.push((packed_base, group.len()));
    }
    bases
}

fn binary_entropy_from_counts(ones: usize, total: usize) -> f64 {
    if total == 0 || ones == 0 || ones == total {
        return 0.0;
    }
    let p = ones as f64 / total as f64;
    let q = 1.0 - p;
    -(p * p.log2()) - (q * q.log2())
}

pub(crate) fn column_order_by_unweighted_entropy(
    base_table: &[(BitVec<usize, Lsb0>, usize)],
) -> Vec<usize> {
    let Some((first_bits, _)) = base_table.first() else {
        return Vec::new();
    };

    let row_count = base_table.len();
    let bit_len = first_bits.len();

    let mut columns_with_entropy: Vec<(usize, f64)> = (0..bit_len)
        .map(|column_idx| {
            let ones = base_table
                .iter()
                .filter(|(bits, _)| bits.get(column_idx).map(|b| *b).unwrap_or(false))
                .count();
            let entropy = binary_entropy_from_counts(ones, row_count);
            (column_idx, entropy)
        })
        .collect();

    columns_with_entropy.sort_by(|(lhs_idx, lhs_entropy), (rhs_idx, rhs_entropy)| {
        lhs_entropy
            .partial_cmp(rhs_entropy)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs_idx.cmp(rhs_idx))
    });

    columns_with_entropy
        .into_iter()
        .map(|(column_idx, _)| column_idx)
        .collect()
}

fn compare_rows_by_column_order(
    lhs: &BitSlice<usize, Lsb0>,
    rhs: &BitSlice<usize, Lsb0>,
    column_order: &[usize],
) -> Ordering {
    for &column_idx in column_order {
        let l = lhs.get(column_idx).map(|bit| *bit).unwrap_or(false);
        let r = rhs.get(column_idx).map(|bit| *bit).unwrap_or(false);
        match r.cmp(&l) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }

    let len = lhs.len().min(rhs.len());
    for idx in 0..len {
        let l = lhs.get(idx).map(|bit| *bit).unwrap_or(false);
        let r = rhs.get(idx).map(|bit| *bit).unwrap_or(false);
        match r.cmp(&l) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }

    lhs.len().cmp(&rhs.len())
}

fn sort_encoding_context(
    base_table: &mut Vec<(BitVec<usize, Lsb0>, usize)>,
    row_to_id: &mut Vec<usize>,
) -> Vec<usize> {
    let base_count = base_table.len();
    if base_count <= 1 {
        return Vec::new();
    }

    let column_order = column_order_by_unweighted_entropy(base_table);

    let mut indices: Vec<usize> = (0..base_count).collect();
    indices.sort_by(|&lhs, &rhs| {
        compare_rows_by_column_order(&base_table[lhs].0, &base_table[rhs].0, &column_order)
            .then_with(|| lhs.cmp(&rhs))
    });

    let mut old_to_new = vec![0usize; base_count];
    for (new_id, old_id) in indices.iter().copied().enumerate() {
        old_to_new[old_id] = new_id;
    }

    *base_table = indices
        .iter()
        .map(|&old_id| base_table[old_id].clone())
        .collect();

    for id in row_to_id.iter_mut() {
        if let Some(&new_id) = old_to_new.get(*id) {
            *id = new_id;
        }
    }

    column_order
}

#[derive(Clone)]
pub struct BaseBitGroups {
    groups: Vec<Vec<usize>>,
    base_bit_mask: BitVec<usize, Lsb0>,
    base_bit_positions: Vec<usize>,
    num_bases: usize,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitGroups")
            .field("groups", &self.groups)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.num_bases)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl BaseBitGroups {
    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let groups = initial_groups(num_rows, 1);
        let base_bit_mask = bitvec![usize, Lsb0; 0; chunk_size];
        let base_bit_positions = Vec::new();
        let num_bits_per_base = 0;
        BaseBitGroups {
            groups,
            base_bit_mask,
            base_bit_positions,
            num_bases: num_bits_per_base,
            num_bits_per_base,
        }
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit position {}", bit_position));

        if self.base_bit_mask[bit_position] {
            return self.num_bases;
        }

        self.base_bit_mask.set(bit_position, true);
        self.num_bits_per_base += 1;
        self.base_bit_positions.push(bit_position);

        let mut new_groups = Vec::new();

        for group in &mut self.groups {
            if group.len() <= 1 {
                continue;
            }

            let mut group_ones = Vec::with_capacity(group.len() / 2 + 1);
            group.retain(|&row| {
                if unsafe { bit_data.get_bit_unchecked(row, bit_position) } {
                    group_ones.push(row);
                    false
                } else {
                    true
                }
            });

            if group.is_empty() {
                *group = group_ones;
            } else if !group_ones.is_empty() {
                new_groups.push(group_ones);
            }
        }

        self.groups.extend(new_groups);
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));
        tracing::trace!(
            "--------------- Adding constant bit positions {:?} --------------",
            bit_positions
        );
        add_constant_bits(
            &mut self.base_bit_mask,
            &mut self.base_bit_positions,
            &mut self.num_bits_per_base,
            bit_positions,
        );
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn get_groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases
    }

    pub fn get_num_bits_per_base(&self) -> usize {
        self.num_bits_per_base
    }

    pub fn get_base_bit_mask(&self) -> &BitSlice<usize, Lsb0> {
        &self.base_bit_mask
    }

    pub fn get_base_bit_positions(&self) -> &[usize] {
        &self.base_bit_positions
    }
}

impl BaseBit for BaseBitGroups {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitGroups::add_bit_position(self, bit_data, bit_position)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitGroups::add_constant_bit_positions(self, bit_positions)
    }

    fn get_encoding_context(&self, bit_data: &BitDataSet, sort: bool) -> EncodingContext {
        let mut base_table = get_selected_bases_from_groups(
            &self.groups,
            &self.base_bit_positions,
            self.num_bases,
            bit_data,
        );
        let mut row_to_id = vec![0usize; bit_data.num_rows()];
        for (id, group) in self.groups.iter().enumerate() {
            for &row in group {
                row_to_id[row] = id;
            }
        }
        let sorted_column_order = if sort {
            Some(sort_encoding_context(&mut base_table, &mut row_to_id))
        } else {
            None
        };
        EncodingContext {
            row_to_id,
            base_table,
            sorted_column_order,
        }
    }

    fn get_num_bases(&self) -> usize {
        BaseBitGroups::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitGroups::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Lsb0> {
        BaseBitGroups::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitGroups::get_base_bit_positions(self)
    }
}

#[derive(Clone)]
pub struct BaseBitHyperLogLogCount {
    base_bit_mask: BitVec<usize, Lsb0>,
    base_bit_positions: Vec<usize>,
    num_bits_per_base: usize,
    num_bases_estimate: usize,
    row_hashes: Vec<u64>,
    bit_hash_words: Vec<u64>,
    registers: Vec<u8>,
}

impl std::fmt::Debug for BaseBitHyperLogLogCount {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitHyperLogLogCount")
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .field("num_bases_estimate", &self.num_bases_estimate)
            .finish()
    }
}

impl BaseBitHyperLogLogCount {
    const HLL_PRECISION: u8 = 8;
    const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let base_bit_mask = bitvec![usize, Lsb0; 0; chunk_size];
        let base_bit_positions = Vec::new();

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut bit_hash_words = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            bit_hash_words.push(Self::splitmix64(&mut state));
        }

        BaseBitHyperLogLogCount {
            base_bit_mask,
            base_bit_positions,
            num_bits_per_base: 0,
            num_bases_estimate: 0,
            row_hashes: vec![0u64; num_rows],
            bit_hash_words,
            registers: vec![0u8; 1usize << Self::HLL_PRECISION],
        }
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(Self::SPLITMIX64_INCREMENT);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn hll_alpha(num_registers: usize) -> f64 {
        match num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => {
                let m = num_registers as f64;
                0.7213 / (1.0 + 1.079 / m)
            }
        }
    }

    fn estimate_from_registers(registers: &[u8], num_rows: usize) -> usize {
        if num_rows == 0 {
            return 0;
        }

        let m = registers.len() as f64;
        let alpha = Self::hll_alpha(registers.len());

        let mut harmonic_sum = 0.0f64;
        let mut zero_count = 0usize;
        for &register in registers {
            if register == 0 {
                zero_count += 1;
            }
            harmonic_sum += 2f64.powi(-(register as i32));
        }

        let mut estimate = alpha * m * m / harmonic_sum;
        if estimate <= 2.5 * m && zero_count > 0 {
            estimate = m * (m / zero_count as f64).ln();
        }

        estimate.round().clamp(1.0, num_rows as f64).max(1.0) as usize
    }

    pub fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit positions {:?}", bit_positions));
        const MIN_ROWS_FOR_PARALLEL: usize = 1_024;

        let new_bit_positions = collect_new_bit_positions(&self.base_bit_mask, bit_positions);
        if new_bit_positions.is_empty() {
            return self.num_bases_estimate;
        }

        let added_count = apply_selected_bit_positions(
            &mut self.base_bit_mask,
            &mut self.base_bit_positions,
            &new_bit_positions,
        );
        self.num_bits_per_base += added_count;

        let stride = bit_data.data.stride;
        self.registers.fill(0);
        let bucket_shift = 64 - Self::HLL_PRECISION as usize;

        debug_assert_eq!(self.row_hashes.len(), bit_data.num_rows());
        debug_assert_eq!(self.bit_hash_words.len(), bit_data.chunk_size());
        debug_assert_eq!(self.registers.len(), 1usize << Self::HLL_PRECISION);

        let new_bit_positions = new_bit_positions.as_slice();

        if self.row_hashes.len() >= MIN_ROWS_FOR_PARALLEL {
            let reduced_registers = self
                .row_hashes
                .par_iter_mut()
                .enumerate()
                .fold(
                    || [0u8; 1usize << Self::HLL_PRECISION],
                    |mut local_registers, (row, row_hash)| {
                        let row_start = row * stride;

                        for &bit_position in new_bit_positions {
                            let bit = unsafe {
                                bit_data
                                    .data
                                    .get_bit_linear_unchecked(row_start + bit_position)
                            } as u64;
                            let hash_word =
                                unsafe { *self.bit_hash_words.get_unchecked(bit_position) };
                            let bit_mask = 0u64.wrapping_sub(bit);
                            *row_hash ^= hash_word & bit_mask;
                        }

                        let bucket = (*row_hash >> bucket_shift) as usize;

                        let suffix = *row_hash << Self::HLL_PRECISION;
                        let rank = (suffix.leading_zeros() as usize + 1)
                            .min((64 - Self::HLL_PRECISION as usize) + 1)
                            as u8;

                        local_registers[bucket] = local_registers[bucket].max(rank);
                        local_registers
                    },
                )
                .reduce(
                    || [0u8; 1usize << Self::HLL_PRECISION],
                    |mut left, right| {
                        for (left_reg, right_reg) in left.iter_mut().zip(right.iter()) {
                            *left_reg = (*left_reg).max(*right_reg);
                        }
                        left
                    },
                );

            self.registers.copy_from_slice(&reduced_registers);
        } else {
            for row in 0..self.row_hashes.len() {
                let row_start = row * stride;

                let row_hash = unsafe { self.row_hashes.get_unchecked_mut(row) };

                for &bit_position in new_bit_positions {
                    let bit = unsafe {
                        bit_data
                            .data
                            .get_bit_linear_unchecked(row_start + bit_position)
                    } as u64;
                    let hash_word = unsafe { *self.bit_hash_words.get_unchecked(bit_position) };
                    let bit_mask = 0u64.wrapping_sub(bit);
                    *row_hash ^= hash_word & bit_mask;
                }

                let bucket = (*row_hash >> bucket_shift) as usize;

                let suffix = *row_hash << Self::HLL_PRECISION;
                let rank = (suffix.leading_zeros() as usize + 1)
                    .min((64 - Self::HLL_PRECISION as usize) + 1) as u8;

                let register = unsafe { self.registers.get_unchecked_mut(bucket) };
                *register = (*register).max(rank);
            }
        }

        self.num_bases_estimate =
            Self::estimate_from_registers(&self.registers, self.row_hashes.len());
        self.num_bases_estimate
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        self.add_bit_positions(bit_data, &[bit_position])
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));

        add_constant_bits(
            &mut self.base_bit_mask,
            &mut self.base_bit_positions,
            &mut self.num_bits_per_base,
            bit_positions,
        );
        self.num_bases_estimate
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases_estimate
    }

    pub fn get_num_bits_per_base(&self) -> usize {
        self.num_bits_per_base
    }

    pub fn get_base_bit_mask(&self) -> &BitSlice<usize, Lsb0> {
        &self.base_bit_mask
    }

    pub fn get_base_bit_positions(&self) -> &[usize] {
        &self.base_bit_positions
    }
}

impl BaseBit for BaseBitHyperLogLogCount {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitHyperLogLogCount::add_bit_position(self, bit_data, bit_position)
    }

    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        BaseBitHyperLogLogCount::add_bit_positions(self, bit_data, bit_positions)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitHyperLogLogCount::add_constant_bit_positions(self, bit_positions)
    }

    fn get_encoding_context(&self, bit_data: &BitDataSet, sort: bool) -> EncodingContext {
        let selected_bit_positions = &self.base_bit_positions;
        let num_rows = self.row_hashes.len();
        let mut hash_to_id: FxHashMap<u64, usize> =
            FxHashMap::with_capacity_and_hasher(num_rows.min(1024), Default::default());
        let mut base_table: Vec<(BitVec<usize, Lsb0>, usize)> = Vec::new();
        let mut row_to_id = Vec::with_capacity(num_rows);

        for (row, &row_hash) in self.row_hashes.iter().enumerate() {
            let id = if let Some(&existing_id) = hash_to_id.get(&row_hash) {
                base_table[existing_id].1 += 1;
                existing_id
            } else {
                let new_id = base_table.len();
                hash_to_id.insert(row_hash, new_id);
                let chunk = unsafe { bit_data.get_chunk_unchecked(row) };
                let mut packed_base = BitVec::with_capacity(selected_bit_positions.len());
                for &bit_pos in selected_bit_positions {
                    packed_base.push(unsafe { *chunk.get_unchecked(bit_pos) });
                }
                base_table.push((packed_base, 1));
                new_id
            };
            row_to_id.push(id);
        }

        let sorted_column_order = if sort {
            Some(sort_encoding_context(&mut base_table, &mut row_to_id))
        } else {
            None
        };

        EncodingContext {
            row_to_id,
            base_table,
            sorted_column_order,
        }
    }

    fn get_num_bases(&self) -> usize {
        BaseBitHyperLogLogCount::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitHyperLogLogCount::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Lsb0> {
        BaseBitHyperLogLogCount::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitHyperLogLogCount::get_base_bit_positions(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::data::{
        BitData, BitDataInfo, BitDataSet, FeatureDataType, FeatureSpec, FeatureTransform,
    };

    fn create_test_bit_data() -> BitDataSet {
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

        let bit_data = BitData {
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

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    #[test]
    fn test_base_bit_hll_count_add_multiple_bits() {
        let bit_data = create_test_bit_data();

        let mut hll_groups =
            BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = hll_groups.add_bit_positions(&bit_data, &[4, 5]);
        assert!(num_bases >= 1);
        assert!(num_bases <= bit_data.num_rows());
        assert_eq!(hll_groups.get_num_bits_per_base(), 2);
    }

    #[test]
    fn test_constant_bits_ignore_duplicates_base_groups() {
        let bit_data = create_test_bit_data();
        let mut groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());

        groups.add_constant_bit_positions(&[1, 1, 2]);

        assert_eq!(groups.get_num_bits_per_base(), 2);
        assert_eq!(groups.get_base_bit_positions(), &[1, 2]);
    }

    #[test]
    fn test_add_bit_positions_ignore_duplicates_hll_groups() {
        let bit_data = create_test_bit_data();
        let mut groups = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());

        let _ = groups.add_bit_positions(&bit_data, &[4, 4, 5]);

        assert_eq!(groups.get_num_bits_per_base(), 2);
        assert_eq!(groups.get_base_bit_positions(), &[4, 5]);
    }

    #[test]
    fn test_base_groups_get_encoding_context_no_sort() {
        let bit_data = create_test_bit_data();
        let mut groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 4);

        let ctx = groups.get_encoding_context(&bit_data, false);

        assert_eq!(ctx.row_to_id.len(), bit_data.num_rows());
        assert!(!ctx.base_table.is_empty());
        let total: usize = ctx.base_table.iter().map(|(_, count)| count).sum();
        assert_eq!(total, bit_data.num_rows());
    }

    #[test]
    fn test_hll_get_encoding_context_no_sort() {
        let bit_data = create_test_bit_data();
        let mut groups = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());
        let _ = groups.add_bit_positions(&bit_data, &[0, 4]);

        let ctx = groups.get_encoding_context(&bit_data, false);

        assert_eq!(ctx.row_to_id.len(), bit_data.num_rows());
        assert!(!ctx.base_table.is_empty());
        let total: usize = ctx.base_table.iter().map(|(_, count)| count).sum();
        assert_eq!(total, bit_data.num_rows());
        // Rows 1 and 5 are identical so they share a base
        assert_eq!(ctx.row_to_id[1], ctx.row_to_id[5]);
    }

    #[test]
    fn test_get_encoding_context_sort_remaps_ids() {
        let bit_data = create_test_bit_data();
        let mut groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 4);

        let unsorted = groups.get_encoding_context(&bit_data, false);
        let sorted = groups.get_encoding_context(&bit_data, true);

        // Same number of bases and same total row count
        assert_eq!(unsorted.base_table.len(), sorted.base_table.len());
        let unsorted_total: usize = unsorted.base_table.iter().map(|(_, c)| c).sum();
        let sorted_total: usize = sorted.base_table.iter().map(|(_, c)| c).sum();
        assert_eq!(unsorted_total, sorted_total);
    }
}
