# Compression baseline and K_min experiment

The validation analysis in `notebooks/v4_inference_Kbased.ipynb` selected
K_min=32 for the next experiment. This is an experimental threshold, not evidence
that modifying compression will improve classification.

## Reproduce the existing-file audit

For the size-only baseline (no K report, validation membership or compression
timings), run:

```powershell
python src/compression/audit_compression.py --output-dir output/compression_baseline_2026-10-30
```

This writes per-image sizes and statistics for all images and each dataset
split, plus a readable `report.md` with the principal results. PNG totals use
only matched, nonempty PNG files; missing PNGs are listed
in the summary. RGB totals include every valid IGD. No image is recompressed.
K columns are omitted when no K report is supplied. The optional K-aware command
below is reserved for subsequent experiments.

From the repository root, using Python 3.9 or newer (standard library only):

```powershell
python src/compression/audit_compression.py --output-dir output/compression_baseline_2026-10-29 --k-report output/cumulative_audit_2026-10-01/per_image.jsonl --k-report-split train --split-file output/splits/cifar10_train_val_0.1.json
```

Choose a new output directory on each run. Inputs and existing outputs are never
overwritten. Default inputs are `data/cifar-10_compressed_1x1` and
`data/cifar-10_resized`, including train/test subdirectories.

Outputs:

- `per_image.csv`: split and image ID, dimensions, grouping, color metadata,
  actual IGD bytes, RGB/RGBA raw bytes, PNG bytes, savings, ratio, bits per pixel,
  and optional K/base width/validation membership.
- `summary.json`: means, population standard deviations, linearly interpolated
  percentiles, totals, expansion counts, missing PNGs and provenance. Groups
  include dataset splits, saved training/validation membership and K<32/K>=32.

The raw reference is width*height*channels bytes (8 bits/channel), before color
transformation. IGD size includes all headers and encoded sections. Savings are
100*(1-IGD/reference); negative values indicate expansion. PNG comparison uses
the existing, already compressed PNG size and is a separate metric. Aggregate
savings use sums of bytes, not an unweighted mean of per-image percentages.

Only the 32-byte IGD header and declared payload length are validated. This is
not a bitstream integrity or lossless round-trip check. Legacy K reports are
trusted after checking dimensions and duplicate IDs; directory names from the
old computer are ignored. Specify their dataset split explicitly. K is left
missing for splits not covered by the report, rather than inferred from names.
The saved train/validation membership should match the checkpoint used for the
classifier analysis; this script does not open model checkpoints.

## Timing and the controlled comparison

Existing IGDs do not preserve compression duration, and the old batch wrapper
does not save per-image durations. File timestamps cannot recover those times.
`compression_seconds` and `compression_timing` are therefore missing/null.
`audit_elapsed_seconds` measures only this audit and must not be reported as
compression time.

Before changing the compressor, benchmark the current executable on a fixed
sample, with an explicit seed, split and sample list. Use the same machine,
thread settings, encoding, grouping and base-counting mode for both versions.
Record executable identity, command, warmup policy, repeats and per-image times.
Keep end-to-end wall time (including process startup, loading and writing)
separate from internal compression time if instrumented. Report median/p95,
variability and aggregate throughput. A fresh benchmark measures current
performance, not historical training-data preparation time.

Then add the optional K_min criterion, preserving the existing selection for
images already meeting it, regenerate features and retrain with the same saved
split and training protocol. Run this audit on the new IGDs in a separate output
directory, with a newly generated K report. Compare size, time, actual achieved
K, affected-image count, and classifier metrics on identical image IDs.
Use paired size/time differences, and leave final test evaluation until the
experiment choices are fixed. No compressor or training changes are part of
the baseline audit.

## Verification

```powershell
python -B -m unittest discover -s src/compression -p test_audit_compression.py
```

Tests cover header parsing/truncation, aggregate byte-weighted savings,
expansion counts, percentiles and portable legacy identifiers.
