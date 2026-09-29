use bitvec::prelude::*;

use super::encoding_core::{BaseTable, CompressedData, DeltaBaseTableData};

use crate::compression::file_format::tags::{
    BASE_TABLE_TAG_DELTA_FIXED, BASE_TABLE_TAG_DELTA_UNARY,
};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

pub struct DeltaEncodeBaseTable {}

#[derive(Debug, Clone, Copy, Default)]
struct DeltaBitStats {
    total_written_bits: usize,
    prefix_bits: usize,
    payload_bits: usize,
    minimally_necessary_bits: usize,
}

impl Filter for DeltaEncodeBaseTable {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Delta encoding base table (post-encoding)");

        let Some(order) = input.entropy_sorted_column_order.clone() else {
            tracing::warn!(
                "DeltaEncodeBaseTable skipped: missing entropy_sorted_column_order; keeping raw base table"
            );
            if !matches!(input.base_table, BaseTable::Raw(_)) {
                input.base_table = BaseTable::Raw(input.base_table.as_raw().to_vec());
            }
            return Ok(input);
        };

        let raw_rows = input.base_table.as_raw().to_vec();
        let num_bases = raw_rows.len();
        if raw_rows.is_empty() {
            input.base_table = BaseTable::Delta(DeltaBaseTableData {
                num_bases: 0,
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: BASE_TABLE_TAG_DELTA_UNARY,
            });
            return Ok(input);
        }

        let row_width = raw_rows[0].0.len();
        if order.iter().any(|&idx| idx >= row_width) {
            tracing::warn!(
                row_width,
                "DeltaEncodeBaseTable skipped: entropy order contains out-of-range column index; keeping raw base table"
            );
            input.base_table = BaseTable::Raw(raw_rows);
            return Ok(input);
        }

        let (first_sort_key, delta_bit_stream, delta_count, delta_stats) =
            match delta_encode_base_rows(&raw_rows, &order, row_width, &get_delta_codec(), 5, true)
            {
                DeltaEncodeResult::Encoded {
                    first_sort_key,
                    delta_bit_stream,
                    delta_count,
                    stats,
                } => (first_sort_key, delta_bit_stream, delta_count, stats),
                DeltaEncodeResult::NotMonotonic => {
                    tracing::warn!(
                        "DeltaEncodeBaseTable skipped: base rows are not monotonic for descending key deltas; keeping raw base table"
                    );
                    input.base_table = BaseTable::Raw(raw_rows);
                    return Ok(input);
                }
                DeltaEncodeResult::ZeroDelta => {
                    tracing::warn!(
                        "DeltaEncodeBaseTable skipped: encountered zero delta (expected unique sorted rows); keeping raw base table"
                    );
                    input.base_table = BaseTable::Raw(raw_rows);
                    return Ok(input);
                }
            };

        let raw_base_table_bits = num_bases * row_width;
        let delta_base_table_bits = first_sort_key.len() + delta_bit_stream.len();

        let quantization_overhead_bits = delta_stats
            .payload_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);
        let total_overhead_bits = delta_stats
            .total_written_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);

        let encoded_id_deviation_bits = input.encoded_data.get_encoded_size();
        let combined_bits = encoded_id_deviation_bits + delta_base_table_bits;
        let base_table_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * delta_base_table_bits as f64 / combined_bits as f64
        };
        let id_deviation_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * encoded_id_deviation_bits as f64 / combined_bits as f64
        };

        let raw_to_delta_ratio = if raw_base_table_bits == 0 {
            0.0
        } else {
            delta_base_table_bits as f64 / raw_base_table_bits as f64
        };

        tracing::debug!(
            raw_base_table_bits,
            delta_base_table_bits,
            delta_count,
            row_width,
            raw_to_delta_ratio,
            "Delta base-table size summary"
        );
        tracing::debug!(
            minimally_necessary_bits = delta_stats.minimally_necessary_bits,
            unary_prefix_overhead_bits = delta_stats.prefix_bits,
            quantization_overhead_bits,
            total_overhead_bits,
            encoded_delta_bits = delta_stats.total_written_bits,
            "Delta coding overhead summary"
        );
        tracing::debug!(
            encoded_base_table_bits = delta_base_table_bits,
            encoded_id_deviation_bits,
            base_table_share_pct,
            id_deviation_share_pct,
            combined_bits,
            "Compressed payload composition (base-table vs id/deviation)"
        );

        input.base_table = BaseTable::Delta(DeltaBaseTableData {
            num_bases,
            first_sort_key,
            delta_bit_stream,
            delta_count,
            sort_column_order: order,
            codec_id: BASE_TABLE_TAG_DELTA_UNARY,
        });
        Ok(input)
    }
}

pub struct DeltaEncodeBaseTableFixed {}

impl Filter for DeltaEncodeBaseTableFixed {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Delta encoding base table with fixed prefix (post-encoding)");

        let Some(order) = input.entropy_sorted_column_order.clone() else {
            tracing::warn!(
                "DeltaEncodeBaseTableFixed skipped: missing entropy_sorted_column_order; keeping raw base table"
            );
            if !matches!(input.base_table, BaseTable::Raw(_)) {
                input.base_table = BaseTable::Raw(input.base_table.as_raw().to_vec());
            }
            return Ok(input);
        };

        let raw_rows = input.base_table.as_raw().to_vec();
        let num_bases = raw_rows.len();
        if raw_rows.is_empty() {
            input.base_table = BaseTable::Delta(DeltaBaseTableData {
                num_bases: 0,
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: BASE_TABLE_TAG_DELTA_FIXED,
            });
            return Ok(input);
        }

        let row_width = raw_rows[0].0.len();
        if order.iter().any(|&idx| idx >= row_width) {
            tracing::warn!(
                row_width,
                "DeltaEncodeBaseTableFixed skipped: entropy order contains out-of-range column index; keeping raw base table"
            );
            input.base_table = BaseTable::Raw(raw_rows);
            return Ok(input);
        }

        const TIER_WIDTHS_FIXED: [usize; 64] = get_delta_codec_fixed();
        let (first_sort_key, delta_bit_stream, delta_count, delta_stats) =
            match delta_encode_base_rows(
                &raw_rows,
                &order,
                row_width,
                &TIER_WIDTHS_FIXED[..63],
                6,
                false,
            ) {
                DeltaEncodeResult::Encoded {
                    first_sort_key,
                    delta_bit_stream,
                    delta_count,
                    stats,
                } => (first_sort_key, delta_bit_stream, delta_count, stats),
                DeltaEncodeResult::NotMonotonic => {
                    tracing::warn!(
                        "DeltaEncodeBaseTableFixed skipped: base rows are not monotonic for descending key deltas; keeping raw base table"
                    );
                    input.base_table = BaseTable::Raw(raw_rows);
                    return Ok(input);
                }
                DeltaEncodeResult::ZeroDelta => {
                    tracing::warn!(
                        "DeltaEncodeBaseTableFixed skipped: encountered zero delta (expected unique sorted rows); keeping raw base table"
                    );
                    input.base_table = BaseTable::Raw(raw_rows);
                    return Ok(input);
                }
            };

        let raw_base_table_bits = num_bases * row_width;
        let delta_base_table_bits = first_sort_key.len() + delta_bit_stream.len();

        let quantization_overhead_bits = delta_stats
            .payload_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);
        let total_overhead_bits = delta_stats
            .total_written_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);

        let encoded_id_deviation_bits = input.encoded_data.get_encoded_size();
        let combined_bits = encoded_id_deviation_bits + delta_base_table_bits;
        let base_table_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * delta_base_table_bits as f64 / combined_bits as f64
        };
        let id_deviation_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * encoded_id_deviation_bits as f64 / combined_bits as f64
        };

        let raw_to_delta_ratio = if raw_base_table_bits == 0 {
            0.0
        } else {
            delta_base_table_bits as f64 / raw_base_table_bits as f64
        };

        tracing::debug!(
            raw_base_table_bits,
            delta_base_table_bits,
            delta_count,
            row_width,
            raw_to_delta_ratio,
            "Delta base-table size summary (fixed prefix)"
        );
        tracing::debug!(
            minimally_necessary_bits = delta_stats.minimally_necessary_bits,
            fixed_prefix_overhead_bits = delta_stats.prefix_bits,
            quantization_overhead_bits,
            total_overhead_bits,
            encoded_delta_bits = delta_stats.total_written_bits,
            "Delta coding overhead summary (fixed prefix)"
        );
        tracing::debug!(
            encoded_base_table_bits = delta_base_table_bits,
            encoded_id_deviation_bits,
            base_table_share_pct,
            id_deviation_share_pct,
            combined_bits,
            "Compressed payload composition (base-table vs id/deviation)"
        );

        input.base_table = BaseTable::Delta(DeltaBaseTableData {
            num_bases,
            first_sort_key,
            delta_bit_stream,
            delta_count,
            sort_column_order: order,
            codec_id: BASE_TABLE_TAG_DELTA_FIXED,
        });
        Ok(input)
    }
}

/// Outcome of the word-level delta encoder. `NotMonotonic`/`ZeroDelta` mirror the
/// two conditions under which the encoder bails out and the caller keeps the raw
/// base table.
enum DeltaEncodeResult {
    Encoded {
        first_sort_key: BitVec<usize, Lsb0>,
        delta_bit_stream: BitVec<usize, Lsb0>,
        delta_count: usize,
        stats: DeltaBitStats,
    },
    NotMonotonic,
    ZeroDelta,
}

/// Delta-encode the base rows by treating each `row_width`-bit sort key as a
/// little-endian array of `usize` words. The descending key deltas are computed
/// word-wise (a few word subtractions per row instead of a bit-by-bit loop), and
/// each delta is emitted via the tiered prefix coder selected by the caller
/// (`tier_widths` / `prefix_bits` / `use_unary_prefix`).
fn delta_encode_base_rows(
    raw_rows: &[(BitVec<usize, Lsb0>, usize)],
    order: &[usize],
    row_width: usize,
    tier_widths: &[usize],
    prefix_bits: usize,
    use_unary_prefix: bool,
) -> DeltaEncodeResult {
    const WORD_BITS: usize = usize::BITS as usize;
    let lb = row_width;
    let num_words = lb.div_ceil(WORD_BITS);
    let max_tier_width = tier_widths.iter().copied().max().unwrap_or(0);
    // The payload buffer must be able to hold a tier whose width exceeds `lb`, so
    // the zero-padded high bits are addressable when emitting the payload.
    let cap_words = lb.max(max_tier_width).div_ceil(WORD_BITS).max(1);

    // Forward permutation: row column -> key bit index. Key bit `lb - 1 - rank`
    // carries row column `order[rank]` (callers guarantee `order[rank] < row_width`).
    let mut key_idx_for_col = vec![usize::MAX; lb];
    for (rank, &col) in order.iter().enumerate() {
        key_idx_for_col[col] = lb - 1 - rank;
    }

    let mut prev = vec![0usize; num_words];
    build_sort_key_words(&raw_rows[0].0, &key_idx_for_col, lb, &mut prev);

    let mut first_sort_key = BitVec::<usize, Lsb0>::from_vec(prev.clone());
    first_sort_key.truncate(lb);

    let mut curr = vec![0usize; num_words];
    let mut delta = vec![0usize; num_words];
    let mut d = vec![0usize; cap_words];

    let mut delta_bit_stream = BitVec::new();
    let mut stats = DeltaBitStats::default();
    let mut delta_count = 0usize;

    for (row, _) in raw_rows.iter().skip(1) {
        build_sort_key_words(row, &key_idx_for_col, lb, &mut curr);

        // delta = prev - curr (a non-zero borrow out means prev < curr).
        let mut borrow = 0usize;
        for w in 0..num_words {
            let (r1, b1) = prev[w].overflowing_sub(curr[w]);
            let (r2, b2) = r1.overflowing_sub(borrow);
            delta[w] = r2;
            borrow = (b1 | b2) as usize;
        }
        if borrow != 0 {
            return DeltaEncodeResult::NotMonotonic;
        }
        if delta.iter().all(|&x| x == 0) {
            return DeltaEncodeResult::ZeroDelta;
        }

        // d = delta - 1, zero-extended into the wider payload buffer.
        d[..num_words].copy_from_slice(&delta);
        for slot in d[num_words..].iter_mut() {
            *slot = 0;
        }
        let mut dec_borrow = 1usize;
        for slot in d[..num_words].iter_mut() {
            let (r, b) = slot.overflowing_sub(dec_borrow);
            *slot = r;
            dec_borrow = b as usize;
            if dec_borrow == 0 {
                break;
            }
        }

        let s = encode_adjusted_delta_words(
            &d,
            lb,
            &mut delta_bit_stream,
            tier_widths,
            prefix_bits,
            use_unary_prefix,
        );
        stats.total_written_bits += s.total_written_bits;
        stats.prefix_bits += s.prefix_bits;
        stats.payload_bits += s.payload_bits;
        stats.minimally_necessary_bits += s.minimally_necessary_bits;

        delta_count += 1;
        std::mem::swap(&mut prev, &mut curr);
    }

    DeltaEncodeResult::Encoded {
        first_sort_key,
        delta_bit_stream,
        delta_count,
        stats,
    }
}

/// Build the sort key for `row` into the little-endian word buffer `out`, scattering
/// each set row bit to its key position via the forward permutation. Iterates only
/// the set bits, so cost is proportional to the row's popcount.
fn build_sort_key_words(
    row: &BitVec<usize, Lsb0>,
    key_idx_for_col: &[usize],
    lb: usize,
    out: &mut [usize],
) {
    const WORD_BITS: usize = usize::BITS as usize;
    out.fill(0);
    for (w, &word) in row.as_raw_slice().iter().enumerate() {
        let base = w * WORD_BITS;
        let mut bits = word;
        while bits != 0 {
            let col = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if col < lb {
                let key_idx = key_idx_for_col[col];
                if key_idx != usize::MAX {
                    out[key_idx / WORD_BITS] |= 1usize << (key_idx % WORD_BITS);
                }
            }
        }
    }
}

/// Number of significant bits in a little-endian word array (0 if the value is 0).
fn word_bit_length(words: &[usize]) -> usize {
    const WORD_BITS: usize = usize::BITS as usize;
    for w in (0..words.len()).rev() {
        if words[w] != 0 {
            return w * WORD_BITS + (WORD_BITS - words[w].leading_zeros() as usize);
        }
    }
    0
}

// Update these from prefix_scheme.py output if you want
pub const fn get_delta_codec() -> [usize; 30] {
    [
        9, 19, 28, 37, 45, 54, 63, 73, 82, 89, 97, 106, 115, 123, 131, 140, 147, 156, 163, 170,
        175, 180, 185, 189, 193, 197, 202, 207, 210, 212,
    ]
}

// Update these from prefix_scheme.py if you want
pub const fn get_delta_codec_fixed() -> [usize; 64] {
    [
        1, 3, 5, 6, 7, 9, 11, 12, 14, 15, 17, 19, 20, 22, 24, 26, 28, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 54, 57, 60, 63, 65, 68, 71, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100,
        104, 107, 111, 115, 119, 123, 128, 133, 140, 145, 150, 156, 162, 170, 177, 187, 197, 212,
    ]
}

// Assign d to the first tier whose width >= bit_length(d); write prefix + d zero-padded.
// No biasing: the optimizer's bit-length model is exact. `d` is a little-endian word
// array holding the adjusted delta in its low `lb` bits (zero-padded above).
fn encode_adjusted_delta_words(
    d: &[usize],
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
    tier_widths: &[usize],
    prefix_bits: usize,
    use_unary_prefix: bool,
) -> DeltaBitStats {
    let d_len = word_bit_length(d);
    let n_active = tier_widths.len();
    let d_bits = d.view_bits::<Lsb0>();

    for (tier, &width) in tier_widths.iter().enumerate() {
        if d_len <= width {
            let p_bits =
                write_delta_prefix(out, tier, n_active, prefix_bits, use_unary_prefix, false);
            out.extend_from_bitslice(&d_bits[..width]);
            return DeltaBitStats {
                minimally_necessary_bits: d_len,
                prefix_bits: p_bits,
                payload_bits: width,
                total_written_bits: p_bits + width,
            };
        }
    }

    // Overflow: write the original d in lb raw bits (not the modified remainder).
    let p_bits = write_delta_prefix(out, n_active, n_active, prefix_bits, use_unary_prefix, true);
    out.extend_from_bitslice(&d_bits[..lb]);
    DeltaBitStats {
        minimally_necessary_bits: d_len,
        prefix_bits: p_bits,
        payload_bits: lb,
        total_written_bits: p_bits + lb,
    }
}

/// Write the tier prefix and return the number of prefix bits written. For unary
/// prefixes a normal tier is `tier` ones then a terminating zero, and overflow is
/// `n_active` ones with no terminator; for fixed prefixes the `prefix_bits`-bit code
/// (tier, or `n_active` for overflow) is written LSB-first.
fn write_delta_prefix(
    out: &mut BitVec<usize, Lsb0>,
    code: usize,
    n_active: usize,
    prefix_bits: usize,
    use_unary_prefix: bool,
    is_overflow: bool,
) -> usize {
    if use_unary_prefix {
        if is_overflow {
            for _ in 0..n_active {
                out.push(true);
            }
            n_active
        } else {
            for _ in 0..code {
                out.push(true);
            }
            out.push(false);
            code + 1
        }
    } else {
        for i in 0..prefix_bits {
            out.push(((code >> i) & 1) == 1);
        }
        prefix_bits
    }
}

// ── Delta decode ─────────────────────────────────────────────────────────────

impl DeltaBaseTableData {
    pub fn decode_rows(&self) -> Result<Vec<(BitVec<usize, Lsb0>, usize)>, EntroGdError> {
        decode_delta_base_rows(
            self.num_bases,
            self.first_sort_key.len(),
            &self.sort_column_order,
            &self.first_sort_key,
            self.delta_count,
            &self.delta_bit_stream,
            self.codec_id,
        )
    }
}

fn decode_delta_base_rows(
    num_bases: usize,
    lb: usize,
    order: &[usize],
    first_sort_key: &BitSlice<usize, Lsb0>,
    delta_count: usize,
    delta_bit_stream: &BitSlice<usize, Lsb0>,
    codec_tag: u8,
) -> Result<Vec<(BitVec<usize, Lsb0>, usize)>, EntroGdError> {
    let _timer = ScopedTimer::debug("Decoding delta-encoded base table rows");
    if num_bases == 0 {
        return Ok(Vec::new());
    }

    if delta_count != num_bases.saturating_sub(1) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "delta_count mismatch: expected {}, got {}",
                num_bases.saturating_sub(1),
                delta_count
            ),
        });
    }

    if order.iter().any(|&idx| idx >= lb) {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta sort column order contains out-of-range index".to_string(),
        });
    }

    // The sort key is an `lb`-bit little-endian integer; treat it as a fixed-width
    // array of `usize` words so each per-row subtraction touches ~lb/word_bits words
    // instead of looping bit-by-bit through bitvec's bounds-checked proxy.
    const WORD_BITS: usize = usize::BITS as usize;
    let num_words = lb.div_ceil(WORD_BITS);
    // Mask for the live bits in the most significant word (keeps bits >= lb at 0 so
    // the word array stays a faithful lb-bit value across subtractions).
    let top_mask: usize = match lb % WORD_BITS {
        0 => usize::MAX,
        rem => (1usize << rem) - 1,
    };

    // Inverse permutation: for each key bit index, the destination row column. Key
    // bit `lb - 1 - rank` carries the row column `order[rank]`.
    let mut col_for_key_idx = vec![0usize; lb];
    for (rank, &col_idx) in order.iter().enumerate() {
        col_for_key_idx[lb.saturating_sub(1 + rank)] = col_idx;
    }

    // prev_key words, reused in place across iterations.
    let mut prev_key = vec![0usize; num_words];
    {
        let src = first_sort_key.to_bitvec();
        let raw = src.as_raw_slice();
        prev_key[..raw.len().min(num_words)].copy_from_slice(&raw[..raw.len().min(num_words)]);
        if let Some(last) = prev_key.last_mut() {
            *last &= top_mask;
        }
    }

    // Scratch buffer for the decoded delta payload, reused across iterations.
    let mut payload_words = vec![0usize; num_words];

    let mut rows = Vec::with_capacity(num_bases);
    rows.push((scatter_key_to_row(&prev_key, &col_for_key_idx, lb), 0usize));

    let mut bit_pos = 0usize;
    for _ in 0..delta_count {
        let payload = decode_adjusted_delta(delta_bit_stream, &mut bit_pos, lb, codec_tag)?;
        // next_key = prev_key - (payload + 1); the +1 is folded into the subtraction's
        // initial borrow. Both operands and the result are `lb`-bit word arrays.
        load_bits_into_words(payload, &mut payload_words);
        subtract_words_plus_one(&mut prev_key, &payload_words, top_mask);
        rows.push((scatter_key_to_row(&prev_key, &col_for_key_idx, lb), 0usize));
    }

    if bit_pos != delta_bit_stream.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta bitstream has trailing/unused bits".to_string(),
        });
    }

    Ok(rows)
}

/// Load a bit payload into a little-endian word buffer, zero-filling unused words.
/// `dst` is sized to the key width (`num_words`); a payload may be wider than that
/// because a tier field width can exceed `lb`, but the encoded value is `delta - 1`
/// (`< 2^lb`), so any bits beyond `dst` are guaranteed zero. We therefore load only
/// the low `dst.len()` words and drop the zero high words.
fn load_bits_into_words(src: &BitSlice<usize, Lsb0>, dst: &mut [usize]) {
    const WORD_BITS: usize = usize::BITS as usize;
    dst.fill(0);
    for (i, chunk) in src.chunks(WORD_BITS).take(dst.len()).enumerate() {
        dst[i] = chunk.load_le::<usize>();
    }
}

/// Compute `key -= payload + 1` in place over little-endian `usize` words. The `+1`
/// is realised by seeding the borrow with 1. `top_mask` clears bits above the key
/// width in the most significant word. Callers must guarantee `key >= payload + 1`
/// (base rows are strictly descending in sort-key order, so this always holds).
fn subtract_words_plus_one(key: &mut [usize], payload: &[usize], top_mask: usize) {
    let mut borrow = 1usize;
    for (k, &p) in key.iter_mut().zip(payload.iter()) {
        let (d1, b1) = k.overflowing_sub(p);
        let (d2, b2) = d1.overflowing_sub(borrow);
        *k = d2;
        borrow = (b1 | b2) as usize;
    }
    if let Some(last) = key.last_mut() {
        *last &= top_mask;
    }
}

/// Reconstruct a row by scattering the key's set bits to their row columns via the
/// precomputed inverse permutation. Reads/writes whole `usize` words and iterates
/// only the set bits of the key, so the cost is proportional to the popcount.
fn scatter_key_to_row(key: &[usize], col_for_key_idx: &[usize], lb: usize) -> BitVec<usize, Lsb0> {
    const WORD_BITS: usize = usize::BITS as usize;
    let mut row = BitVec::repeat(false, lb);
    let row_words = row.as_raw_mut_slice();
    for (w, &word) in key.iter().enumerate() {
        let base = w * WORD_BITS;
        let mut bits = word;
        while bits != 0 {
            let key_idx = base + bits.trailing_zeros() as usize;
            let col = col_for_key_idx[key_idx];
            row_words[col / WORD_BITS] |= 1usize << (col % WORD_BITS);
            bits &= bits - 1;
        }
    }
    row
}

fn decode_adjusted_delta<'a>(
    bits: &'a BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
    codec_tag: u8,
) -> Result<&'a BitSlice<usize, Lsb0>, EntroGdError> {
    if codec_tag == BASE_TABLE_TAG_DELTA_UNARY {
        decode_adjusted_delta_unary(bits, bit_pos, lb)
    } else if codec_tag == BASE_TABLE_TAG_DELTA_FIXED {
        decode_adjusted_delta_fixed(bits, bit_pos, lb)
    } else {
        Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported delta codec tag {}", codec_tag),
        })
    }
}

fn decode_adjusted_delta_unary<'a>(
    bits: &'a BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<&'a BitSlice<usize, Lsb0>, EntroGdError> {
    const TIER_WIDTHS: [usize; 30] = get_delta_codec();
    let n_active = TIER_WIDTHS.len();

    let mut tier = 0usize;
    while *bit_pos < bits.len() && bits[*bit_pos] {
        tier += 1;
        *bit_pos += 1;
        if tier == n_active {
            break;
        }
    }

    if tier < n_active {
        if *bit_pos >= bits.len() || bits[*bit_pos] {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid delta prefix terminator".to_string(),
            });
        }
        *bit_pos += 1;
    }

    let payload_width = if tier < n_active {
        TIER_WIDTHS[tier]
    } else {
        lb
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    Ok(payload)
}

fn decode_adjusted_delta_fixed<'a>(
    bits: &'a BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<&'a BitSlice<usize, Lsb0>, EntroGdError> {
    const TIER_WIDTHS: [usize; 64] = get_delta_codec_fixed();
    const PREFIX_BITS: usize = 6;
    // 63 normal tiers (0..62); tier 63 = 0b111111 = all-ones in 6 bits = overflow sentinel.
    const N_ACTIVE: usize = TIER_WIDTHS.len() - 1;

    if *bit_pos + PREFIX_BITS > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "insufficient bits for delta tier ID".to_string(),
        });
    }

    let mut tier = 0usize;
    for i in 0..PREFIX_BITS {
        if bits[*bit_pos + i] {
            tier |= 1usize << i;
        }
    }
    *bit_pos += PREFIX_BITS;

    if tier > N_ACTIVE {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta tier ID out of range".to_string(),
        });
    }

    let payload_width = if tier < N_ACTIVE {
        TIER_WIDTHS[tier]
    } else {
        lb
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    Ok(payload)
}
