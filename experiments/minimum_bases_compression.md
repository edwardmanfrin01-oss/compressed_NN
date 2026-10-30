# Minimum dictionary size experiment

The optional image-compression argument `--k-min 32` first runs the unchanged
adaptive selector, then checks the actual number of distinct bases. Below the
minimum, it adds remaining bit positions in ascending entropy order, one at a
time, accepting size-estimate regressions. It stops at the first configuration
with at least the requested number of bases. A single addition may overshoot.
There is no further size optimization. If all positions are exhausted, the
maximum attainable distinction is retained and a diagnostic is printed.

The minimum is checked exactly even with `--base-counting approximate`.
Without `--k-min`, the new filter is a no-op and the existing pipeline is used.
The IGD format is unchanged. Feature extraction is unchanged.

## Manual steps

These commands are instructions; the dataset batch has not been run as part of
this implementation. Run them from the repository root.

Build the compressor and the streaming IGD dictionary-count helper:

```powershell
cargo build --release --locked --manifest-path src/gdcompress/Cargo.toml --bins
```

Focused synthetic tests (not executed during implementation):

```powershell
cargo test --locked --manifest-path src/gdcompress/Cargo.toml --test minimum_bases
```

They cover the disabled/sufficient cases, exact enforcement with both counting
modes, stopping at the first qualifying partition including overshoot,
unreachable thresholds and rejection of zero.

Prepare the two dataset splits:

```powershell
python src/compression/batch_compression.py --k-min 32 --input-dir data/cifar-10_resized/train
python src/compression/batch_compression.py --k-min 32 --input-dir data/cifar-10_resized/test
```

Default output directories are `data/cifar-10_compressed_1x1_Kmin/train` and
`data/cifar-10_compressed_1x1_Kmin/test`. Baselines default to the corresponding
split in `data/cifar-10_compressed_1x1`. Use `--output-dir` and `--existing-dir`
to override them. Keep baseline compression settings consistent with the batch
defaults (1x1, precise counting, Huffman, base-table compression enabled).

The helper reads actual dictionary lengths without reconstructing pixels.
Existing IGDs at or above the minimum are copied byte-for-byte. Below-threshold
images are compressed from their original PNGs. Missing baseline IGDs are also
compressed from PNGs. Baseline inputs are never overwritten. Nonmatching pixel
grouping is rejected. The new output directory contains a complete split.

Each invocation writes a separate `compression_actions_*.csv` recording original
and achieved K, action, requested threshold and whether it was reached. Standard
error is captured and displayed only on compression failure. Unreachable
thresholds remain recorded in the CSV. No per-image
timing instrumentation is added in this change.

Existing output files are rejected by default. `--resume` skips existing files;
use it only for an interrupted run with the same settings. Choose a different
output directory when changing the threshold or compression configuration.

For a single image:

```powershell
.\src\gdcompress\target\release\gdcompress.exe data/cifar-10_resized/train/image_00001.png --pixel-grouping 1x1 --k-min 32 -o tmp/image_00001.kmin32.igd
```
