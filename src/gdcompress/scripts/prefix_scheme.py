#!/usr/bin/env python3
"""Analyze tiered prefix+payload schemes from delta bit-length histograms.

The script finds the best (mode, prefix_bits, tier_count) configuration for a
given bit-length histogram. The per-configuration search uses exact dynamic
programming, so within each family the result is provably optimal — the only
remaining uncertainty is whether the outer search bounds are wide enough.
Each family table prints a saturation note (interior vs boundary).

Encoding model:
- Bit lengths are sorted and partitioned into contiguous tiers.
- Each tier's payload width = the maximum bit_len in its bin range
  (fixed-width within tier). Contiguous tiers are without loss of generality:
  a non-contiguous tier could only increase its payload width.
- Prefix length is determined by mode:
    * unary: tier i uses i+1 bits, with the last tier omitting the
      terminator (truncated unary). Pass --unary-include-terminator to keep
      a terminator on the last tier too.
    * fixed: every tier uses a constant number of prefix bits.

Input CSV format must include at least:
  bit_len,count
Extra columns are ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TierSegment:
    tier_index: int
    bit_min: int
    bit_max: int
    count: int


@dataclass(frozen=True)
class PlanResult:
    mode: str  # "unary" or "fixed"
    prefix_bits: int  # fixed prefix bits, 0 for unary
    n_tiers: int
    total_bits: int
    overhead_pct: float
    avg_bits_per_delta: float
    payload_maxima: Tuple[int, ...]
    segments: Tuple[TierSegment, ...]


FamilyKey = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find approximate optimal tiered prefix encodings from bit-length histograms."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=(
            "target/base_table_delta_compression/"
            "joel-filipe-187166.png__sorted__unsigned__cs-SrgbWithLinearAlpha"
            "__cm-YCoCgR__pg-4__gt-ForFirstPixel.csv"
        ),
        help="Path to histogram CSV with bit_len,count columns.",
    )
    parser.add_argument(
        "--max-unary-tiers",
        type=int,
        default=32,
        help="Maximum unary tier count to evaluate.",
    )
    parser.add_argument(
        "--max-fixed-prefix-bits",
        type=int,
        default=8,
        help="Evaluate fixed prefixes from 1..this value.",
    )
    parser.add_argument(
        "--knee-min-improvement-pct",
        type=float,
        default=0.20,
        help=(
            "Knee threshold in percent: when incremental improvement drops "
            "below this percent, extra tiers are likely not worth it."
        ),
    )
    parser.add_argument(
        "--complexity-penalty-pct",
        type=float,
        default=0.05,
        help=(
            "Percent of lower-bound bits to penalize per tier in "
            "complexity-aware ranking. Higher -> prefers fewer tiers."
        ),
    )
    parser.add_argument(
        "--unary-include-terminator",
        action="store_true",
        help=(
            "Use the old unary model where the last tier also carries a "
            "terminator bit. Default is truncated unary (last tier saves "
            "one bit)."
        ),
    )
    parser.add_argument(
        "--max-payload-bits",
        type=int,
        default=None,
        help=(
            "Cap the maximum tier payload width at this many bits. "
            "Deltas needing more bits are folded into this width and handled "
            "by the overflow tier. Default: no cap (use full histogram range)."
        ),
    )
    parser.add_argument(
        "--show-segments-for",
        type=str,
        default="best",
        choices=["none", "best", "knee", "all"],
        help="Print tier coverage segments for selected solutions.",
    )
    return parser.parse_args()


def load_histogram(path: Path) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"bit_len", "count"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            bit_len = int(row["bit_len"])
            count = int(row["count"])
            if count <= 0:
                continue
            histogram[bit_len] = histogram.get(bit_len, 0) + count

    if not histogram:
        raise ValueError("Histogram is empty after parsing counts.")
    return histogram


def build_prefix_sums(values: Sequence[int]) -> List[int]:
    out = [0]
    s = 0
    for v in values:
        s += v
        out.append(s)
    return out


def range_sum(prefix: Sequence[int], left: int, right: int) -> int:
    return prefix[right + 1] - prefix[left]


def optimize_for_tiers(
    bit_lens: Sequence[int],
    count_prefix: Sequence[int],
    n_tiers: int,
    mode: str,
    fixed_prefix_bits: int,
    unary_include_terminator: bool,
) -> Tuple[int, Tuple[int, ...], Tuple[TierSegment, ...]]:
    n = len(bit_lens)
    if n_tiers < 1 or n_tiers > n:
        raise ValueError("n_tiers must be between 1 and number of distinct bit lengths")

    def prefix_cost(tier_idx: int) -> int:
        if mode == "unary":
            if not unary_include_terminator and tier_idx == n_tiers - 1:
                return n_tiers - 1
            return tier_idx + 1
        return fixed_prefix_bits

    def group_cost(i: int, j: int, tier_idx: int) -> int:
        payload_bits = bit_lens[j]
        group_count = range_sum(count_prefix, i, j)
        return (prefix_cost(tier_idx) + payload_bits) * group_count

    inf = math.inf
    dp: List[List[float]] = [[inf] * n for _ in range(n_tiers)]
    prev: List[List[int]] = [[-1] * n for _ in range(n_tiers)]

    for j in range(n):
        dp[0][j] = group_cost(0, j, 0)

    for k in range(1, n_tiers):
        for j in range(k, n):
            best = inf
            best_i = -1
            for i in range(k, j + 1):
                c = dp[k - 1][i - 1] + group_cost(i, j, k)
                if c < best:
                    best = c
                    best_i = i
            dp[k][j] = best
            prev[k][j] = best_i

    boundaries: List[Tuple[int, int]] = []
    j = n - 1
    for k in range(n_tiers - 1, -1, -1):
        i = prev[k][j] if k > 0 else 0
        boundaries.append((i, j))
        j = i - 1
    boundaries.reverse()

    payloads: List[int] = []
    segments: List[TierSegment] = []
    for tier_idx, (i, j) in enumerate(boundaries):
        payloads.append(bit_lens[j])
        segments.append(
            TierSegment(
                tier_index=tier_idx,
                bit_min=bit_lens[i],
                bit_max=bit_lens[j],
                count=range_sum(count_prefix, i, j),
            )
        )

    return int(dp[n_tiers - 1][n - 1]), tuple(payloads), tuple(segments)


def evaluate_candidates(
    bit_lens: Sequence[int],
    counts: Sequence[int],
    lb_bits: int,
    total_count: int,
    max_unary_tiers: int,
    max_fixed_prefix_bits: int,
    unary_include_terminator: bool,
) -> Dict[FamilyKey, List[PlanResult]]:
    count_prefix = build_prefix_sums(counts)
    n = len(bit_lens)
    families: Dict[FamilyKey, List[PlanResult]] = {}

    def add_plan(mode: str, prefix_bits: int, tiers: int) -> None:
        bits, payloads, segments = optimize_for_tiers(
            bit_lens,
            count_prefix,
            tiers,
            mode=mode,
            fixed_prefix_bits=prefix_bits if mode == "fixed" else 0,
            unary_include_terminator=unary_include_terminator,
        )
        plan = PlanResult(
            mode=mode,
            prefix_bits=prefix_bits if mode == "fixed" else 0,
            n_tiers=tiers,
            total_bits=bits,
            overhead_pct=100.0 * (bits / lb_bits - 1.0),
            avg_bits_per_delta=bits / total_count,
            payload_maxima=payloads,
            segments=segments,
        )
        families.setdefault((mode, plan.prefix_bits), []).append(plan)

    unary_limit = min(max_unary_tiers, n)
    for tiers in range(1, unary_limit + 1):
        add_plan("unary", 0, tiers)

    for pb in range(1, max_fixed_prefix_bits + 1):
        fixed_limit = min(1 << pb, n)
        for tiers in range(1, fixed_limit + 1):
            add_plan("fixed", pb, tiers)

    return families


def describe_mode(plan: PlanResult) -> str:
    if plan.mode == "unary":
        return "unary"
    return f"fixed/{plan.prefix_bits}b"


def family_label(key: FamilyKey) -> str:
    mode, pb = key
    return "unary" if mode == "unary" else f"fixed-{pb}b"


def family_tier_limit(key: FamilyKey, n_distinct: int, args: argparse.Namespace) -> int:
    mode, pb = key
    if mode == "unary":
        return min(args.max_unary_tiers, n_distinct)
    return min(1 << pb, n_distinct)


def find_knee(family: Sequence[PlanResult], min_improvement_pct: float) -> PlanResult:
    if not family:
        raise ValueError("empty family")
    ordered = sorted(family, key=lambda x: x.n_tiers)
    knee = ordered[0]
    for idx in range(1, len(ordered)):
        prev = ordered[idx - 1]
        cur = ordered[idx]
        if prev.total_bits <= 0:
            knee = prev
            break
        gain_pct = 100.0 * (prev.total_bits - cur.total_bits) / prev.total_bits
        if gain_pct < min_improvement_pct:
            knee = prev
            break
        knee = cur
    return knee


def complexity_aware_best(
    families: Dict[FamilyKey, List[PlanResult]],
    lb_bits: int,
    penalty_pct_per_tier: float,
) -> PlanResult:
    penalty_bits_per_tier = lb_bits * (penalty_pct_per_tier / 100.0)
    best: Optional[Tuple[float, PlanResult]] = None
    for family in families.values():
        for p in family:
            score = p.total_bits + penalty_bits_per_tier * p.n_tiers
            if best is None or score < best[0]:  # type: ignore[index]
                best = (score, p)
    assert best is not None
    return best[1]


def shannon_entropy(counts: Iterable[int]) -> float:
    cs = list(counts)
    total = sum(cs)
    if total == 0:
        return 0.0
    h = 0.0
    for c in cs:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------- printing ----------


def print_header(input_path: Path, histogram: Dict[int, int]) -> Tuple[int, int]:
    total = sum(histogram.values())
    lb_bits = sum(bit_len * c for bit_len, c in histogram.items())
    min_bit = min(histogram)
    max_bit = max(histogram)
    distinct = len(histogram)
    mean = lb_bits / total
    entropy = shannon_entropy(histogram.values())

    print(f"Input CSV          : {input_path}")
    print(f"Distinct bit-lens  : {distinct}  (range {min_bit}..{max_bit})")
    print(f"Total deltas       : {total:,}")
    print(f"Mean bit-len       : {mean:.3f} bits/delta")
    print(f"Bit-len entropy    : {entropy:.3f} bits over the bin distribution")
    print(f"Lower bound        : {lb_bits:,} bits  ({lb_bits / 8.0:,.1f} bytes)")
    print()
    print(
        "DP is exact per (mode, prefix_bits, k). The outer search is exhaustive\n"
        "over the configured ranges, so the reported best is the global optimum\n"
        "unless a BOUNDARY HIT warning fires below."
    )
    print()

    return total, lb_bits


FAMILY_HEADER = "  {:<3} {:>4} {:>14} {:>10} {:>9} {:>9}  {}"
FAMILY_ROW = "  {:<3} {:>4} {:>14,} {:>9.2f}% {:>9.3f} {:>9}  {}"
FAMILY_COLS = ("tag", "k", "bits", "vs-lb", "avg/Δ", "Δ-prev", "payload maxima")


def fmt_delta_prev(curr: PlanResult, prev: Optional[PlanResult]) -> str:
    if prev is None or prev.total_bits == 0:
        return "—"
    pct = 100.0 * (curr.total_bits - prev.total_bits) / prev.total_bits
    return f"{pct:+.2f}%"


def print_family(
    key: FamilyKey,
    family: Sequence[PlanResult],
    tier_limit: int,
    knee: PlanResult,
    raise_flag_name: str,
) -> None:
    ordered = sorted(family, key=lambda x: x.n_tiers)
    best = min(ordered, key=lambda x: x.total_bits)
    label = family_label(key)
    print(f"Family: {label}  (k = 1..{tier_limit}, {len(ordered)} configurations)")
    print(FAMILY_HEADER.format(*FAMILY_COLS))
    for idx, p in enumerate(ordered):
        prev = ordered[idx - 1] if idx > 0 else None
        tag = ""
        if p is best:
            tag += "*"
        if p is knee and p is not best:
            tag += "+"
        print(
            FAMILY_ROW.format(
                tag,
                p.n_tiers,
                p.total_bits,
                p.overhead_pct,
                p.avg_bits_per_delta,
                fmt_delta_prev(p, prev),
                list(p.payload_maxima),
            )
        )
    if best.n_tiers < tier_limit:
        print(
            f"  -> best at k={best.n_tiers}; tested up to k={tier_limit} "
            f"(interior, converged)."
        )
    else:
        print(
            f"  -> best at k={best.n_tiers} = tier limit; "
            f"BOUNDARY HIT -- raise {raise_flag_name} to confirm optimum."
        )
    print("  legend: * = family best   + = knee")
    print()


def print_recommendation(
    best_by_bits: PlanResult,
    best_by_complexity: PlanResult,
    best_unary_knee: Optional[PlanResult],
    best_fixed_knee: Optional[PlanResult],
) -> None:
    print("Recommendation:")
    rec_fmt = "  {:<22} {:<10} k={:>2}  bits={:>14,}  vs-lb={:>+7.2f}%  payload={}"

    def row(label: str, p: PlanResult) -> str:
        return rec_fmt.format(
            label,
            describe_mode(p),
            p.n_tiers,
            p.total_bits,
            p.overhead_pct,
            list(p.payload_maxima),
        )

    print(row("pure-bit optimum", best_by_bits))
    print(row("complexity-aware", best_by_complexity))
    if best_unary_knee is not None:
        print(row("best unary knee", best_unary_knee))
    if best_fixed_knee is not None:
        print(row("best fixed knee", best_fixed_knee))
    print()


def unary_prefix_string(tier_idx: int, n_tiers: int, include_terminator: bool) -> str:
    if n_tiers == 1:
        return "(none)"
    if tier_idx == n_tiers - 1 and not include_terminator:
        return "1" * tier_idx
    return "1" * tier_idx + "0"


def fixed_prefix_string(tier_idx: int, prefix_bits: int) -> str:
    return format(tier_idx, f"0{prefix_bits}b")


def print_plan_segments(plan: PlanResult, include_terminator: bool) -> None:
    print(
        f"Tier segments for {describe_mode(plan)} with {plan.n_tiers} tiers "
        f"(bits={plan.total_bits:,}, vs-lb={plan.overhead_pct:+.2f}%):"
    )
    print(
        "  {:>4} {:>10} {:>8} {:>8} {:>12} {:>14}".format(
            "tier", "prefix", "bit_min", "bit_max", "count", "tier bits"
        )
    )
    for seg in plan.segments:
        if plan.mode == "unary":
            pref = unary_prefix_string(seg.tier_index, plan.n_tiers, include_terminator)
            pref_bits = 0 if pref == "(none)" else len(pref)
        else:
            pref = fixed_prefix_string(seg.tier_index, plan.prefix_bits)
            pref_bits = plan.prefix_bits
        tier_bits = (pref_bits + seg.bit_max) * seg.count
        print(
            "  {:>4} {:>10} {:>8} {:>8} {:>12,} {:>14,}".format(
                seg.tier_index,
                pref,
                seg.bit_min,
                seg.bit_max,
                seg.count,
                tier_bits,
            )
        )
    print()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)

    histogram = load_histogram(input_path)

    total_count, lb_bits = print_header(input_path, histogram)

    bit_lens = sorted(histogram)
    counts = [histogram[b] for b in bit_lens]

    families = evaluate_candidates(
        bit_lens=bit_lens,
        counts=counts,
        lb_bits=lb_bits,
        total_count=total_count,
        max_unary_tiers=args.max_unary_tiers,
        max_fixed_prefix_bits=args.max_fixed_prefix_bits,
        unary_include_terminator=args.unary_include_terminator,
    )
    if not families:
        raise RuntimeError("No plans generated. Check your input and limits.")

    knees: Dict[FamilyKey, PlanResult] = {
        key: find_knee(fam, args.knee_min_improvement_pct)
        for key, fam in families.items()
    }

    n_distinct = len(bit_lens)
    family_order: List[FamilyKey] = []
    if ("unary", 0) in families:
        family_order.append(("unary", 0))
    family_order.extend(sorted(k for k in families if k[0] == "fixed"))

    for key in family_order:
        tier_limit = family_tier_limit(key, n_distinct, args)
        raise_flag = (
            "--max-unary-tiers" if key[0] == "unary" else "--max-fixed-prefix-bits"
        )
        print_family(key, families[key], tier_limit, knees[key], raise_flag)

    all_plans = [p for fam in families.values() for p in fam]
    best_by_bits = min(all_plans, key=lambda p: p.total_bits)
    best_by_complexity = complexity_aware_best(
        families, lb_bits, args.complexity_penalty_pct
    )
    best_unary_knee = knees.get(("unary", 0))
    fixed_knees = [knees[k] for k in knees if k[0] == "fixed"]
    best_fixed_knee = (
        min(fixed_knees, key=lambda p: p.total_bits) if fixed_knees else None
    )

    print_recommendation(
        best_by_bits,
        best_by_complexity,
        best_unary_knee,
        best_fixed_knee,
    )

    include_term = args.unary_include_terminator
    printed = set()

    def show(plan: Optional[PlanResult]) -> None:
        if plan is None:
            return
        key = (plan.mode, plan.prefix_bits, plan.n_tiers)
        if key in printed:
            return
        printed.add(key)
        print_plan_segments(plan, include_term)

    if args.show_segments_for in ("best", "all"):
        show(best_by_bits)
    if args.show_segments_for in ("knee", "all"):
        show(best_unary_knee)
        show(best_fixed_knee)


if __name__ == "__main__":
    main()
