# V4 — MSB-first bit order interleaved across channels

V4 retains the v3 adaptive formula: a_r=log2(1+d_r)/log2(1+d_max), q_r=floor(S_i*a_r), z_r=r+sum(q_0..q_r). S_i is the largest integer that fits in uint16 and is recalculated using the new deltas. The first delta is zero; a single base produces a zero map.

The only change to feature construction is the bit permutation before numerically sorting the dictionary: 7,15,23,6,14,22,...,0,8,16, filtered to the selected positions. Indices are zero-based and the original bytes are LSB-first. Base columns are permuted, rows are sorted again, rank/delta are recalculated, and spatial IDs are remapped. The number of bases, frequencies, and pixel-to-base associations are preserved. Three 8-bit channels and 1x1 groups are supported.

Compression and color conversion are unchanged. The example files use the YCoCgR color model. No new IGDs or intermediate JSON files on disk are required. Each NPZ contains z_u16 and metadata.json, including original/new order, color model, number of bases, binary extrema after permutation, maximum delta, scale, and z_max.

## Usage from the project root (PowerShell)

```powershell
cargo build --release --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target --bin igd_extract
python src/batch_network_features.py --version v4 --input-dir data/cifar-10_compressed_1x1/train --output-dir output/v4_cumulative_rank_interleaved/train
python src/batch_network_features.py --version v4 --input-dir data/cifar-10_compressed_1x1/test --output-dir output/v4_cumulative_rank_interleaved/test
```

Adjust paths to the actual directories. The batch reads IGDs directly from the specified directory, without recursion. Add --verify to check decoded chunks and --resume to resume an export of the same version. It produces scales_v4.csv and summary_v4.jsonl (one row per image) for each processed directory; summaries are rebuilt even on resume and replaced only upon completion. No full dictionary or spatial matrix is duplicated in the summary.

For a single file: igd_extract INPUT.igd --representation cumulative-rank-interleaved -o OUTPUT.npz.

The v2v3_ResNet50, valid_v2v3_ResNet50, and v2v3_inference notebooks support v4. Set CONFIG["version"]="v4", scale=None, and checkpoint_name=None for training, maintaining label manifests according to the existing procedure. Use a new kernel and statistics calculated on the v4 training set; do not reuse v3 statistics/checkpoints. The model architecture is unchanged. The notebook default remains v3.

## Subsequent experiments (not implemented)

- MSB-first within each channel without interleaving: 7,6,...,0,15,14,...,8,23,22,...,16.
- RGB / YCoCgR / YCbCr comparison with explicit, documented transformations, avoiding applying YCoCgR to YCbCr values packed as RGB.

The existing order used by previous versions remains available. No new training has been started and the full dataset has not been exported.
