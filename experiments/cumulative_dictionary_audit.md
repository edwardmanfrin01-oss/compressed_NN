# Preliminary analysis: rank + cumulative deltas

## Results from October 1, 2026

Analyzed 24,517 1x1 IGDs with zero errors. There are 50,000 available originals:
25,483 compressed files are missing from the analyzed directory. The other
directory, `data/cifar-10_compressed`, adds no missing image names.
Detailed reports: `output/cumulative_audit_2026-10-01/` (excluded from Git).

Number of bases: minimum 2, median 932, 95th percentile approximately 5,331,
99th percentile approximately 14,285, maximum 18,805 in `image_06405.igd`.

| S | Maximum z | Images exceeding uint16 |
|---|---:|---:|
| 1 | 26,882 | 0 |
| 2 | 39,953 | 0 |
| 3 | 45,068 | 0 |
| 4 | 57,505 | 0 |
| 5 | 65,074 | 0 |
| 6 | 76,540 | 23 |

Largest common integer S for uint16: 5. The maximum z at S=5 occurs in
`image_14525.igd`, which differs from the image with the most bases. No common
S fits uint8: even at S=0, 18,280 images have a maximum rank above 255.
There are 2,060 dictionaries with equal gaps; these are not automatically
the worst cases because the number of bases also matters.

Conclusion: trying S=1,2,3 with uint16 is reasonable. S=5 fits the analyzed
files but leaves only 461 levels of headroom; it provides no guarantee for
new images or missing files. The future generator must check for overflow.
No new maps were generated and no training was started.

This analysis does not create the new representation or modify compressed
files. It reads every available IGD in the specified directory, one image at
a time. Results apply only to those files: comparing names against PNG images
reports coverage without establishing that contents and parameters match.

## Running from the project root

```powershell
cargo build --manifest-path src/igd_extract/Cargo.toml --bin analyze_dictionaries --target-dir src/gdcompress/target --release --offline --locked
python src/analyze_cumulative_dataset.py --output-dir output/cumulative_audit
```

The output directory must be new. Options: `--input-dir`, `--original-dir`,
`--helper`. Defaults: `data/cifar-10_compressed_pg_1x1` and `data/cifar-10_resized`.
The Rust reader uses the original codecs; Python handles scanning and summaries.
No external Python libraries are required. No intermediate spatial JSONs are created.

## Definition

Full bases (with constants reinserted) are sorted using the same convention as
the previous experiment: increasing bit positions, with the first character
treated as the MSB to define the numerical value. This is not RGB luminance
ordering. 1x1 groups and bases up to 32 bits are supported; other cases are
reported as errors.

For K sorted bases, consider the K-1 positive gaps. The initial delta is zero.

```
a[j] = log2(1 + delta[j]) / log2(1 + max_delta)
q[j] = round_half_up(S * a[j])
c_last = sum(q)
z_max = K - 1 + c_last
```

Normalization uses float64 and summation uses integers; there is no frequency
weighting. The largest admissible S is found with binary search, including S=0.
For a single-base dictionary, z is always zero: S has no finite upper bound.
Numerical results use the rounding convention actually implemented by the code.
Spatial IDs are not checked: this analyzes dictionaries rather than fully
verifying the decoder.

## Output

- `per_image.jsonl`: K, bits/base, min/max gap, mean log-normalized gap,
  equal-gap indicator, z_max for S=0..16, maximum S for uint8/uint16.
  The final cumulative sum is z_max - (K-1).
- `summary.json`: coverage, errors, K distribution (nearest-rank quantiles
  calculated with a rounded index), worst-case images, overflow by S, and
  largest common S. Interpret null using the `unbounded` and
  `impossible_even_at_s0` fields.
- `missing_igd.txt`: original image names without an available compressed file.

The base count includes every row in the stored dictionary. Limits apply to
analyzed files and are not guaranteed for future images. Existing training
inputs and outputs remain unchanged.

## Verification

```powershell
cargo test --manifest-path src/igd_extract/Cargo.toml --bin analyze_dictionaries --target-dir src/gdcompress/target --release --offline --locked
```

Arithmetic tests: equally spaced gaps, rounding threshold, impossibility due
to rank, and a single-base dictionary. An additional check was performed on
three IGDs in `tmp/manual-test`: comparison against previously exported JSON
dictionaries, independent Python calculation of all 17 maxima, and verification
that the maximum S fits uint16 while S+1 exceeds it.
