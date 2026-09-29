use crate::compression::base_bits::{BaseBit, BaseBitGroups, BaseBitHyperLogLogCount};
use crate::compression::data::BitDataSet;
use crate::compression::entropy::{ConstantBitPolarity, EntropyScoredContext};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;
use fxhash::FxHashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompressedSizeBreakdown {
    pub total_size: usize,
    pub num_bases: usize,
    pub size_bases: usize,
    pub size_deviations: usize,
    pub size_ids: usize,
    pub size_params: usize,
}

pub(crate) fn calculate_compressed_size_breakdown<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> CompressedSizeBreakdown {
    let n = bit_data.num_rows();
    let d = bit_data.num_features();
    let chunk_size = bit_data.chunk_size();
    let n_b = base_bit_groups.get_num_bases();

    let len_b = base_bit_groups.get_num_bits_per_base();
    let len_d = chunk_size - len_b;

    let len_id = if n_b == 0 {
        0
    } else {
        bits_needed_nonzero(n_b)
    };

    let size_bases = n_b * len_b;
    let size_deviations = n * len_d;
    let size_ids = n * len_id;
    let size_dev_bits = chunk_size;
    let size_params = 16 * d + 16 + size_dev_bits; // should revisit this
    let total_size = size_bases + size_deviations + size_ids + size_params;

    CompressedSizeBreakdown {
        total_size,
        num_bases: n_b,
        size_bases,
        size_deviations,
        size_ids,
        size_params,
    }
}

pub(crate) fn calculate_compressed_size<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> usize {
    calculate_compressed_size_breakdown(bit_data, base_bit_groups).total_size
}

fn format_base_bit_mask<B: BaseBit + ?Sized>(base_bit_groups: &B) -> String {
    base_bit_groups
        .get_base_bit_mask()
        .iter()
        .map(|b| if *b { "1" } else { "0" })
        .collect()
}

struct SelectBasesCsvLogger {
    writer: Option<BufWriter<File>>,
}

impl SelectBasesCsvLogger {
    fn new(output_path: Option<&Path>) -> Self {
        let Some(path) = output_path else {
            return Self { writer: None };
        };

        let selected_path = Self::next_available_path(path);
        let file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&selected_path)
        {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(path = %selected_path.display(), ?error, "failed to create select-bases debug csv");
                return Self { writer: None };
            }
        };

        let mut writer = BufWriter::new(file);
        if let Err(error) = writeln!(
            writer,
            "bit_position,num_bases,size_bases,size_deviations,size_ids,size_params,total_size"
        ) {
            tracing::warn!(path = %selected_path.display(), ?error, "failed to write select-bases csv header");
            return Self { writer: None };
        }

        if selected_path != path {
            tracing::info!(
                requested_path = %path.display(),
                selected_path = %selected_path.display(),
                "select-bases debug csv path already existed; using incremented suffix"
            );
        }

        Self {
            writer: Some(writer),
        }
    }

    fn next_available_path(path: &Path) -> PathBuf {
        if !path.exists() {
            return path.to_path_buf();
        }

        let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "select_bases_debug".to_string());
        let extension = path.extension().map(|e| e.to_string_lossy().into_owned());

        for suffix in 1..=usize::MAX {
            let file_name = match &extension {
                Some(ext) => format!("{}_{}.{}", stem, suffix, ext),
                None => format!("{}_{}", stem, suffix),
            };
            let candidate = parent.join(file_name);
            if !candidate.exists() {
                return candidate;
            }
        }

        path.to_path_buf()
    }

    fn log_row(&mut self, bit_position: usize, breakdown: CompressedSizeBreakdown) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };

        if let Err(error) = writeln!(
            writer,
            "{},{},{},{},{},{},{}",
            bit_position,
            breakdown.num_bases,
            breakdown.size_bases,
            breakdown.size_deviations,
            breakdown.size_ids,
            breakdown.size_params,
            breakdown.total_size
        ) {
            tracing::warn!(
                ?error,
                "failed to write select-bases csv row; disabling csv logging"
            );
            self.writer = None;
        }
    }
}

pub struct SelectBases {
    pub patience: usize,
}

pub struct BaseSelectionContext {
    pub bit_data: BitDataSet,
    pub base_bits: Box<dyn BaseBit>,
    pub constant_bit_polarity: ConstantBitPolarity,
}

impl BaseSelectionContext {
    pub fn new(
        bit_data: BitDataSet,
        base_bits: Box<dyn BaseBit>,
        constant_bit_polarity: ConstantBitPolarity,
    ) -> Self {
        Self {
            bit_data,
            base_bits,
            constant_bit_polarity,
        }
    }
}

fn merge_constant_bit_positions(constant_bit_polarity: &ConstantBitPolarity) -> Vec<usize> {
    let mut constant_positions = Vec::with_capacity(
        constant_bit_polarity.constant_zero_bit_positions.len()
            + constant_bit_polarity.constant_one_bit_positions.len(),
    );
    constant_positions.extend_from_slice(&constant_bit_polarity.constant_zero_bit_positions);
    constant_positions.extend_from_slice(&constant_bit_polarity.constant_one_bit_positions);
    constant_positions.sort_unstable();
    constant_positions.dedup();
    constant_positions
}

/// Profiling-oriented selector that adds every bit position as base bits.
///
/// By default (`split_into_batches = 1`), all positions are added in a single
/// batch call. Set `split_into_batches > 1` to split the full set of bit
/// positions into that many smaller additions.
pub struct SelectBasesProfileAllBits {
    /// Desired number of additions used to add all bit positions.
    ///
    /// - `0` and `1` both behave as a single batch add.
    /// - Values larger than the number of bit positions are clamped.
    pub split_into_batches: usize,
    pub base_bit_impl: BaseBitImpl,
}

pub struct SelectBasesDebug {
    pub patience: usize,
    pub debug_csv_paths: Mutex<Vec<PathBuf>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseBitImpl {
    Naive,
    HyperLogLogCount,
}

impl Filter for SelectBases {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        } = input;
        let constant_positions = merge_constant_bit_positions(&constant_bit_polarity);
        let base_bit_groups = select_base_bits(
            &bit_data,
            entropy_scores,
            &constant_positions,
            self.patience,
        );
        Ok(BaseSelectionContext::new(
            bit_data,
            Box::new(base_bit_groups),
            constant_bit_polarity,
        ))
    }
}

impl Filter for SelectBasesProfileAllBits {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting all base bits for profiling (split_into_batches={}, base_bit_impl={:?})",
            self.split_into_batches, self.base_bit_impl
        ));

        let EntropyScoredContext {
            bit_data,
            constant_bit_polarity,
            ..
        } = input;
        let chunk_size = bit_data.chunk_size();
        let all_bit_positions: Vec<usize> = (0..chunk_size).collect();
        let mut base_bit_groups: Box<dyn BaseBit> = match self.base_bit_impl {
            BaseBitImpl::Naive => Box::new(BaseBitGroups::new(bit_data.num_rows(), chunk_size)),
            BaseBitImpl::HyperLogLogCount => Box::new(BaseBitHyperLogLogCount::new(
                bit_data.num_rows(),
                chunk_size,
            )),
        };

        if !all_bit_positions.is_empty() {
            let requested_batches = self.split_into_batches.max(1);
            let batch_count = requested_batches.min(all_bit_positions.len());

            if batch_count == 1 {
                base_bit_groups
                    .as_mut()
                    .add_bit_positions(&bit_data, &all_bit_positions);
            } else {
                let batch_size = all_bit_positions.len().div_ceil(batch_count);
                for batch in all_bit_positions.chunks(batch_size) {
                    base_bit_groups.as_mut().add_bit_positions(&bit_data, batch);
                }
            }
        }

        let compressed_size = calculate_compressed_size(&bit_data, base_bit_groups.as_ref());
        tracing::info!(
            selected_num_bases = base_bit_groups.get_num_bases(),
            selected_num_bits_per_base = base_bit_groups.get_num_bits_per_base(),
            selected_mask = %format_base_bit_mask(base_bit_groups.as_ref()),
            selected_compressed_size_bytes = compressed_size / 8,
            "selected base bit mask (profile all bits, compressed size in bytes)"
        );

        Ok(BaseSelectionContext::new(
            bit_data,
            base_bit_groups,
            constant_bit_polarity,
        ))
    }
}

impl Filter for SelectBasesDebug {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with CSV debug and patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        } = input;
        let constant_positions = merge_constant_bit_positions(&constant_bit_polarity);

        let selected_debug_csv_path = {
            let guard = self
                .debug_csv_paths
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.first().cloned()
        };

        if selected_debug_csv_path.is_none() {
            tracing::warn!(
                "SelectBasesDebug has no remaining debug CSV paths; running without CSV logging"
            );
        }

        let base_bit_groups = select_base_bits_debug(
            &bit_data,
            entropy_scores,
            &constant_positions,
            self.patience,
            selected_debug_csv_path.as_deref(),
        );

        if let Some(selected_path) = selected_debug_csv_path {
            let mut guard = self
                .debug_csv_paths
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(pos) = guard.iter().position(|p| *p == selected_path) {
                guard.remove(pos);
            }
        }

        Ok(BaseSelectionContext::new(
            bit_data,
            Box::new(base_bit_groups),
            constant_bit_polarity,
        ))
    }
}

fn select_base_bits(
    bit_data: &BitDataSet,
    mut entropy: Vec<(usize, f64)>,
    constant_positions: &[usize],
    patience: usize,
) -> BaseBitGroups {
    let mut non_improving_count = 0usize;
    let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );
    base_bit_groups.add_constant_bit_positions(constant_positions);

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for &(bit_position, _) in entropy.iter().skip(constant_positions.len()) {
        if best_base_bit_groups.get_num_bits_per_base()
            >= (bit_data.chunk_size() as f64 * 1.0) as usize
        {
            break;
        }

        trial_base_bit_groups.add_bit_position(bit_data, bit_position);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        tracing::debug!(
            bit_position,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base bit"
        );

        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            non_improving_count = 0;
        } else {
            non_improving_count += 1;
        }

        if non_improving_count >= patience {
            break;
        }
    }
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %format_base_bit_mask(&best_base_bit_groups),
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (compressed size in bytes)"
    );
    best_base_bit_groups
}

fn select_base_bits_debug(
    bit_data: &BitDataSet,
    mut entropy: Vec<(usize, f64)>,
    constant_positions: &[usize],
    patience: usize,
    debug_csv_path: Option<&Path>,
) -> BaseBitGroups {
    let mut non_improving_count = 0usize;
    let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
    let mut csv_logger = SelectBasesCsvLogger::new(debug_csv_path);

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );
    for &bit_position in constant_positions {
        base_bit_groups.add_constant_bit_positions(&[bit_position]);
        let breakdown = calculate_compressed_size_breakdown(bit_data, &base_bit_groups);
        csv_logger.log_row(bit_position, breakdown);
    }

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for &(bit_position, _) in entropy.iter().skip(constant_positions.len()) {
        if best_base_bit_groups.get_num_bits_per_base()
            >= (bit_data.chunk_size() as f64 * 1.0) as usize
        {
            break;
        }

        trial_base_bit_groups.add_bit_position(bit_data, bit_position);
        let trial_breakdown = calculate_compressed_size_breakdown(bit_data, &trial_base_bit_groups);
        csv_logger.log_row(bit_position, trial_breakdown);
        let trial_compressed_size = trial_breakdown.total_size;

        tracing::debug!(
            bit_position,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base bit"
        );

        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            non_improving_count = 0;
        } else {
            non_improving_count += 1;
        }

        if non_improving_count >= patience {
            break;
        }
    }
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %format_base_bit_mask(&best_base_bit_groups),
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (debug csv, compressed size in bytes)"
    );
    best_base_bit_groups
}

pub struct SelectBasesThreshold {
    pub patience: usize,
    pub base_bit_impl: BaseBitImpl,
    pub entropy_threshold: f64,
}

impl Filter for SelectBasesThreshold {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits in batches with patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        } = input;
        let constant_positions = merge_constant_bit_positions(&constant_bit_polarity);
        let base_bit_groups = match self.base_bit_impl {
            BaseBitImpl::Naive => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                &constant_positions,
                self.entropy_threshold,
                self.patience,
            ),
            BaseBitImpl::HyperLogLogCount => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                &constant_positions,
                self.entropy_threshold,
                self.patience,
            ),
        };
        Ok(BaseSelectionContext::new(
            bit_data,
            base_bit_groups,
            constant_bit_polarity,
        ))
    }
}

fn select_base_bits_threshold_optimized(
    bit_data: &BitDataSet,
    mut base_bit_groups: impl BaseBit + Clone + 'static,
    mut entropy: Vec<(usize, f64)>,
    constant_positions: &[usize],
    entropy_threshold: f64,
    patience: usize,
) -> Box<dyn BaseBit> {
    let mut non_improving_count = 0usize;

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );

    let threshold_bits: Vec<usize> = entropy
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val <= entropy_threshold)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    let num_threshold_bits = threshold_bits.len();

    base_bit_groups.add_constant_bit_positions(constant_positions);

    let low_entropy_bits: Vec<usize> = threshold_bits
        .iter()
        .copied()
        .filter(|pos| !constant_positions.contains(pos))
        .collect();
    if !low_entropy_bits.is_empty() {
        base_bit_groups.add_bit_positions(bit_data, &low_entropy_bits);
    }

    tracing::info!(
        num_threshold_bits,
        entropy_threshold,
        "bulk-added bits at or below entropy threshold"
    );

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for &(bit_position, _) in entropy.iter().skip(num_threshold_bits) {
        trial_base_bit_groups.add_bit_positions(bit_data, &[bit_position]);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        tracing::debug!(
            bit_position,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base-bit"
        );

        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            non_improving_count = 0;
        } else {
            non_improving_count += 1;
        }

        if non_improving_count >= patience {
            break;
        }
    }
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %format_base_bit_mask(&best_base_bit_groups),
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (threshold optimized, compressed size in bytes)"
    );
    Box::new(best_base_bit_groups)
}

pub struct SelectBasesAdaptive {
    /// Geometric width-decay factor for entropy clustering. Must be in (0, 1).
    /// `0.5` → first cluster spans half the entropy range, then 1/4, 1/8, …
    pub width_decay: f64,
    /// Patience for the naive single-bit fallback after a cluster fails.
    pub patience: usize,
    pub base_bit_impl: BaseBitImpl,
}

impl Filter for SelectBasesAdaptive {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Adaptive entropy-clustered selection (width_decay={}, patience={})",
            self.width_decay, self.patience
        ));
        assert!(
            self.width_decay > 0.0 && self.width_decay < 1.0,
            "width_decay must be in (0, 1), got {}",
            self.width_decay
        );

        let EntropyScoredContext {
            bit_data,
            entropy_scores,
            constant_bit_polarity,
        } = input;
        let constant_positions = merge_constant_bit_positions(&constant_bit_polarity);

        let base_bits = match self.base_bit_impl {
            BaseBitImpl::Naive => select_base_bits_adaptive(
                &bit_data,
                BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                &constant_positions,
                self.width_decay,
                self.patience,
            ),
            BaseBitImpl::HyperLogLogCount => select_base_bits_adaptive(
                &bit_data,
                BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                &constant_positions,
                self.width_decay,
                self.patience,
            ),
        };

        Ok(BaseSelectionContext::new(
            bit_data,
            base_bits,
            constant_bit_polarity,
        ))
    }
}

/// Greedy left-to-right sweep on entropy-sorted bits.
///
/// Returns cluster sizes (in order); their sum equals `sorted.len()`. The
/// first cluster spans up to `width_decay * (e_max - e_min)` of entropy from
/// its starting value; the allowed width is multiplied by `width_decay` after
/// each closed cluster. The first cluster is therefore the largest and later
/// clusters become progressively more sensitive, degenerating to size 1 once
/// `width` drops below the smallest gap between consecutive values.
fn cluster_by_entropy_width(sorted: &[(usize, f64)], width_decay: f64) -> Vec<usize> {
    if sorted.is_empty() {
        return Vec::new();
    }
    let e_min = sorted.first().unwrap().1;
    let e_max = sorted.last().unwrap().1;
    let range = e_max - e_min;
    if range == 0.0 {
        return vec![sorted.len()];
    }

    let mut sizes = Vec::new();
    let mut i = 0;
    let mut width = range * width_decay;
    while i < sorted.len() {
        let start_e = sorted[i].1;
        let mut j = i + 1;
        while j < sorted.len() && sorted[j].1 - start_e <= width {
            j += 1;
        }
        sizes.push(j - i);
        i = j;
        width *= width_decay;
    }
    sizes
}

/// Two-phase entropy-clustered selector.
///
/// Phase A: cluster the entropy-sorted bits by `width_decay`, add a cluster at
/// a time from the current best state — on improvement keep it, on failure
/// roll back and proceed to Phase B.
///
/// Phase B: naive single-bit additions on the remaining bits, with `patience`
/// consecutive non-improving attempts before stopping.
fn select_base_bits_adaptive<B: BaseBit + Clone + 'static>(
    bit_data: &BitDataSet,
    mut base_bit_groups: B,
    mut entropy: Vec<(usize, f64)>,
    constant_positions: &[usize],
    width_decay: f64,
    patience: usize,
) -> Box<dyn BaseBit> {
    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    base_bit_groups.add_constant_bit_positions(constant_positions);

    let constant_set: FxHashSet<usize> = constant_positions.iter().copied().collect();
    let non_const: Vec<(usize, f64)> = entropy
        .into_iter()
        .filter(|(pos, _)| !constant_set.contains(pos))
        .collect();

    let mut best = base_bit_groups;
    let mut best_size = calculate_compressed_size(bit_data, &best);

    // Phase A — entropy-clustered batched additions.
    let cluster_sizes = cluster_by_entropy_width(&non_const, width_decay);
    let mut cursor = 0usize;
    let mut phase_a_clusters_committed = 0usize;
    for cluster_size in cluster_sizes {
        let end = cursor + cluster_size;
        let block: Vec<usize> = non_const[cursor..end].iter().map(|(p, _)| *p).collect();

        let mut trial = best.clone();
        trial.add_bit_positions(bit_data, &block);
        let trial_size = calculate_compressed_size(bit_data, &trial);

        tracing::debug!(
            cluster_size,
            cursor,
            trial_size,
            best_size,
            "evaluated entropy cluster"
        );

        if trial_size < best_size {
            best = trial;
            best_size = trial_size;
            cursor = end;
            phase_a_clusters_committed += 1;
        } else {
            break;
        }
    }

    tracing::info!(
        phase_a_clusters_committed,
        phase_a_bits_committed = cursor,
        phase_a_size_bytes = best_size / 8,
        "adaptive phase A complete"
    );

    // Phase B — naive single-bit fallback on the remainder.
    let mut fails = 0usize;
    let mut phase_b_bits_committed = 0usize;
    let mut trial = best.clone();
    for (bit_pos, _) in non_const[cursor..].iter() {
        trial.add_bit_position(bit_data, *bit_pos);
        let trial_size = calculate_compressed_size(bit_data, &trial);

        tracing::debug!(
            bit_pos = *bit_pos,
            trial_size,
            best_size,
            "evaluated single bit (phase B)"
        );

        if trial_size < best_size {
            best = trial.clone();
            best_size = trial_size;
            fails = 0;
            phase_b_bits_committed += 1;
        } else {
            fails += 1;
            if fails >= patience {
                break;
            }
        }
    }

    tracing::info!(
        selected_num_bases = best.get_num_bases(),
        selected_num_bits_per_base = best.get_num_bits_per_base(),
        selected_mask = %format_base_bit_mask(&best),
        selected_compressed_size_bytes = best_size / 8,
        phase_b_bits_committed,
        "adaptive entropy-clustered selection complete"
    );

    Box::new(best)
}

#[cfg(test)]
mod adaptive_clustering_tests {
    use super::cluster_by_entropy_width;

    #[test]
    fn empty_input_returns_empty() {
        let sizes = cluster_by_entropy_width(&[], 0.5);
        assert!(sizes.is_empty());
    }

    #[test]
    fn all_equal_entropies_single_cluster() {
        let data: Vec<(usize, f64)> = (0..10).map(|i| (i, 0.42)).collect();
        let sizes = cluster_by_entropy_width(&data, 0.5);
        assert_eq!(sizes, vec![10]);
    }

    #[test]
    fn uniform_entropies_halve_geometrically() {
        // entropies 0.0, 1.0, 2.0, ..., 15.0 → range 15.
        // decay=0.5 → widths 7.5, 3.75, 1.875, ...
        let data: Vec<(usize, f64)> = (0..16).map(|i| (i, i as f64)).collect();
        let sizes = cluster_by_entropy_width(&data, 0.5);
        assert_eq!(sizes.iter().sum::<usize>(), 16);
        // First cluster should be the largest.
        for i in 1..sizes.len() {
            assert!(sizes[0] >= sizes[i], "first cluster must dominate");
        }
        // Once width drops below 1.0, every subsequent cluster is size 1.
        assert_eq!(*sizes.last().unwrap(), 1);
    }

    #[test]
    fn skewed_distribution_captures_dense_region_first() {
        // 90 bits with low entropy ~0, 10 bits scattered in (0.5, 1.0].
        let mut data: Vec<(usize, f64)> = (0..90).map(|i| (i, 0.001 * i as f64)).collect();
        for i in 0..10 {
            data.push((90 + i, 0.5 + 0.05 * i as f64));
        }
        data.sort_by(|a, b| a.1.total_cmp(&b.1));
        let sizes = cluster_by_entropy_width(&data, 0.5);
        assert_eq!(sizes.iter().sum::<usize>(), 100);
        // First cluster should swallow the dense low-entropy region.
        assert!(
            sizes[0] >= 90,
            "first cluster should capture dense region (got {})",
            sizes[0]
        );
    }
}
