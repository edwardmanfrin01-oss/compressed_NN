# Extracting bases, IDs, and bit positions from IGD

This program reads an `.igd` using the local Rust library `../gdcompress`
and exports indented JSON. It does not rerun compression or modify the
compressor repository. The format and APIs match the examined local copy,
commit `8927486e4ab16a33ab872fe1e530514653a01b4d`.

## Usage (the command below explains why the igd_extract executable is in the gdcompress project directory)

Run from the project directory `C:\Users\edward\CODING_PROJECTS\AARHUS`:

```powershell
cargo run --release --locked --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target -- data/cifar-10_compressed/image_00000.igd -o output/igd/image_00000.json --verify
```

After building, you can use the executable directly:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00001.igd -o output/igd/image_00001.json --verify
```

Options:

- `-o PATH` / `--output PATH`: destination; defaults to `INPUT.decoded.json`.
- `--network-features`: directly generates the first experiment's representation:
  sorted dictionary, exact deltas, and spatial rank and delta maps.
- `--network-npz`: directly generates the .npz file containing the two spatial maps
  (`spatial_ranks` and `spatial_deltas`), ready as model input. Values are normalized
  and mapped to [0,255] to obtain a uint8 representation (see below).
- `--verify`: reconstructs transformed chunks from bases, IDs, the map, and
  deviations, and compares **every bit of every sample** with the `gdcompress`
  decoder. It does not create an RGB image; this check requires extra work and memory.
- `--include-deviations`: also saves deviations, which are required to fully
  reconstruct transformed chunks from the JSON alone.
- `--help`: summary.

Destination directories are created as needed. Existing output is not overwritten.
Errors produce a message on stderr and exit code 1. The program checks the container
length before passing it to the parser: a file interrupted during writing may be incomplete.

To include deviations, use a different name if the first output already exists:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00000.igd -o output/igd/image_00000.with_deviations.json --include-deviations --verify
```

## Representation for the first network experiment

`--network-features` reads the IGD directly and writes a separate JSON; generic
JSON is not saved as an intermediate file. Example for images compressed with
`--pixel-grouping 1x1`:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00000.igd -o output/network/image_00000.network.json --network-features --verify
```

Each base is a fixed-width binary string. The extractor sorts it in ascending
lexicographic order, interpreting the leftmost character as the most significant
bit under the experimental convention. This avoids converting long bases to
`float` or `u64`, which would lose bits.

The new ID is the **rank** of the sorted base. The first delta is zero; each
subsequent delta is the exact difference from the preceding base by rank. JSON
preserves both `delta_bits` and `spatial_delta_bits` as exact binary strings,
and two normalized matrices for use as the network's first two channels:

```text
spatial_rank_normalized[y][x] = rank / (num_bases - 1)
spatial_delta_log2_normalized[y][x] = log2(delta + 1) / log2(max_delta + 1)
```

Both channels use zero when the denominator would be zero.
`spatial_delta_bits[y][x]` does not describe a difference from a neighboring
pixel: it is the delta of that pixel's base in the sorted dictionary. Geometry
is retained in matrix position `(y, x)`; the precise meaning of each delta
remains in `sorted_dictionary`.

Frequencies are not multiplied by deltas: each sorted base stores its
`frequency`, equal to the number of times its rank occurs in `spatial_rank_ids`.
Multiplying these values would erase this distinction.

Main fields added by the `igd-network-features-v1` format:

| Field | Meaning |
|---|---|
| `sorted_dictionary` | Sorted base, original ID, rank, exact delta, and frequency |
| `base_min_bits` / `base_max_bits` | Exact extrema of the sorted dictionary |
| `spatial_rank_ids` | New IDs/ranks in the original image geometry |
| `spatial_delta_bits` | Exact delta associated with each rank, arranged spatially |
| `spatial_rank_normalized` | First numerical channel for the network |
| `spatial_delta_log2_normalized` | Second numerical channel for the network |
| `representation` | Full ordering and normalization conventions |

With `1x1` grouping, a 224x224 image produces two 224x224 maps, one per channel.
JSON files are intended for checking and traceability: for full-dataset training,
use `--network-npz`, which directly writes two `uint8` channels without creating
intermediate JSON:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed_pg_1x1/image_00000.igd -o output/network/image_00000.network.npz --network-npz --verify
```

Conversion to `uint8` is performed only after normalization:

```text
rank_u8  = round(255 * spatial_rank_normalized)
delta_u8 = round(255 * spatial_delta_log2_normalized)
```

The PyTorch loader approximately restores both channels to `[0, 1]` using
`array.astype(float32) / 255.0`. The NPZ contains `rank_u8.npy` and `delta_u8.npy`,
both shaped `(height, width)`, plus `metadata.json`. For a 224x224 image,
the tensors alone occupy exactly `224 * 224 * 2 = 100352` bytes.

To process the dataset without ever writing intermediate generic JSON, run
compression followed by export from the project root:

```powershell
python src/batch_compression.py
python src/batch_network_features.py
```

The first script uses `1x1` by default and writes to
`data/cifar-10_compressed_pg_1x1`; the second reads that directory and writes to
`output/network_features_1x1_v1`. Both refuse to overwrite existing files.
After an interruption, use `--resume`. First try a few images by specifying
`--input-dir`, `--output-dir`, and, for the exporter, `--verify`.

Alternatively, `python src/build_1x1_npz_dataset.py` performs
PNG -> IGD 1x1 -> NPZ uint8 in the same iteration and retains the IGD.
It creates no dataset JSON; use `--verify` only for a small initial sample
and `--resume` to continue after an interruption.

## The map that prevents mixing up bits

**All JSON indices and IDs start at 0.** Bit strings are character sequences;
they must not be interpreted as MSB-first binary numbers.

The main relationship is:

```text
transformed_chunk[base_bit_positions[k]] = bases[id].bits[k]
```

Schematic example:

```json
{
  "base_bit_positions": [0, 1, 4],
  "bases": [{"id": 0, "bits": "101"}],
  "sample_base_ids": [0]
}
```

The first character of `101` goes to position 0, the second to position 1,
and the third to position 4. Other positions are deviations. The list is sorted
by increasing position and base strings are exported in the same order:
there is no need to know which position the algorithm selected first.

The **selection history**, for example `1 -> 2 -> 5 -> 3`, is not stored in
the IGD file. Therefore, `selection_order` is `null`. This is distinct from
the entropy-based column order used internally by delta coding: the decoder
undoes that permutation before export. Recording the actual history would
require instrumenting the selection stage during a new compression run.

## JSON fields

| Field | Meaning |
|---|---|
| `schema` | Export schema version: `igd-extract-v1` |
| `source` | Path, size, IGD/EGD versions, and encodings of the input file |
| `conventions` | Explicit conventions for indices, strings, and layout |
| `image` | Dimensions, channels, color model, transformation, grouping, and grid |
| `features` | Per channel/feature: chunk offset, bit count, type, and transformation |
| `chunk_bits` | Actual bit count of the transformed chunk |
| `base_bits` | Length of `bases[id].bits`, including constant bits |
| `variable_base_bits` | Length of the variable part of the dictionary only |
| `base_bit_positions` | Full-base column -> chunk position mapping |
| `variable_base_bit_positions` | `variable_bits` column -> chunk position mapping |
| `constant_zero_bit_positions` | Constant positions (value 0) reinserted into full bases |
| `constant_one_bit_positions` | Constant positions (value 1) reinserted into full bases |
| `deviation_bit_positions` | Deviation column -> chunk position mapping |
| `bases` | Dictionary: ID, full string, variable string, frequency |
| `sample_base_ids` | One ID per pixel group, in original spatial order |
| `sample_deviation_bits` | One string per group, or `null` if not requested |
| `selection_order` | `null`: history is not present in the format |
| `verification` | Whether comparison with the decoder was performed and passed |

IDs retain the dictionary numbering in the file. Frequencies are recalculated
by counting sample IDs, even for a delta-coded dictionary. The number of bases
and their lengths can vary by image. An ID is meaningful only together with
the dictionary of the same image.

The grid is traversed from left to right and top to bottom:

```text
group_x = sample_index % image.grid_width
group_y = sample_index // image.grid_width
pixel_x_start = group_x * image.pixel_group_width
pixel_y_start = group_y * image.pixel_group_height
```

Edge groups may extend beyond image dimensions: these dimensions remain in
JSON to allow correct cropping. The extractor rejects condensed samples with
weights, which do not provide a reliable original spatial map.

## Caution: RGB chunks and transformed chunks

For RGB8, a raw 3x3 group contains `3 * 9 * 8 = 216` bits. With the
`ForFirstPixel` transformation, each channel stores the first value in 8 bits
and 8 differences encoded in 9 bits:

```text
3 * (8 + 8 * 9) = 240 bit
```

This is the case for the examined real file `image_00000.igd`. Positions
therefore refer to a 240-bit transformed chunk with three 80-bit features,
rather than 216 RGB bits. Bits are grouped by channel; numerical fields are
stored least-significant-bit first. To obtain RGB pixels, first reconstruct
the chunk and then invert the group and color transformations using metadata
and the `gdcompress` functions.

**Bases + IDs + positions generally do not contain all original values.**
Full reconstruction also requires deviations. Without them, known bits can
be identified, but missing bits must not be replaced with zeros while claiming
the result is the original reconstruction. JSON also cannot recover information
lost before compression, for example during image resizing.

## Reading in Python

Standard library only; run from the AARHUS directory:

```python
import json
from pathlib import Path

data = json.loads(Path("output/igd/image_00000.json").read_text(encoding="utf-8"))
sample_index = 0
base_id = data["sample_base_ids"][sample_index]
base = data["bases"][base_id]

# Full chunk map, with None where deviations are missing.
chunk = [None] * data["chunk_bits"]
for position, bit in zip(data["base_bit_positions"], base["bits"], strict=True):
    chunk[position] = int(bit)

if data["includes_deviations"]:
    deviation = data["sample_deviation_bits"][sample_index]
    for position, bit in zip(data["deviation_bit_positions"], deviation, strict=True):
        chunk[position] = int(bit)
    assert None not in chunk

print("ID:", base_id)
print("Base:", base["bits"])
print("Positions:", data["base_bit_positions"])
print("Transformed chunk:", chunk)
```

For NumPy, IDs can be arranged on the grid using
`np.asarray(data['sample_base_ids']).reshape(data['image']['grid_height'], data['image']['grid_width'])`.
This is a grid of **group-local IDs**, not a normal RGB image.

## Code verification

```powershell
cargo test --release --locked --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target
```

Tests compare chunks reconstructed from serialized JSON against pre-compression
data: normal/RLE/Huffman encodings, raw/delta-unary/delta-fixed dictionaries,
Raw/ForFirstPixel/ForMin transformations, constant 0 and 1 bits, selection in
non-increasing order, and edge groups. Truncated files and refusal to overwrite
are also checked.

`Cargo.lock` retains dependency versions resolved in the local copy; keep it
to reproduce the build. JSON prioritizes readability and may be much larger
than the IGD, especially when including deviations. It is not yet the optimized
input format for a neural network.
# Cumulative + rank mode (`uint16`)

## Versioned batch conversion

From the project root:

```powershell
python src/batch_network_features.py --version v1 --resume
python src/batch_network_features.py --version v2 --scale 1 --resume
```

The default input is `data/cifar-10_compressed_1x1`. V1 saves two uint8 channels
to `output/network_features_1x1_v1`; v2 saves `z_u16` to
`output/cumulative_rank_s1` (the name changes with S). Without `--version`,
v1 is used; for v2, S defaults to 1. `--scale` is invalid with v1.
You can specify `--input-dir`, `--output-dir`, `--extractor-bin`, and `--verify`.
Both versions retain the `image_XXXXX.network.npz` names.

`--resume` checks ZIP integrity, required arrays, representation, and S before
skipping existing output; with `--verify`, it also requires the previous output
to have been verified. It does not compare IGD contents against the previously
used file: after recompression, use a new directory. The first error stops the
batch and displays the reason; completed files can be reused with `--resume`.

This mode retains the previous two-channel `--network-npz` output.
For the new experiment, from the AARHUS root:

```powershell
cargo build --manifest-path src/igd_extract/Cargo.toml --bin igd_extract --target-dir src/gdcompress/target --release --offline --locked
src/gdcompress/target/release/igd_extract.exe data/cifar-10_compressed_1x1/image_00000.igd --representation cumulative-rank --scale 1 --verify -o output/cumulative_rank_s1/image_00000.npz
```

`--scale` is an integer in 0..65535, defaults to 1, and is only usable with
`--representation cumulative-rank`. S=0 produces rank only. 1x1 groups are
required. Without `-o`, the name is `INPUT.cumulative-s1.npz` (including S).
Existing files are not overwritten. uint16 overflow stops the operation
before file creation: no clipping is applied.

For sorted bases, calculate `d[0]=0`, `d[r]=base[r]-base[r-1]`, then
`q[r]=round(S*log2(1+d[r])/log2(1+d_max))`, rounding halves upward.
If d_max=0, all q values are zero. The cumulative sum of q is added to the
original, unnormalized rank. Finally, each pixel receives the value for its
base. Gaps are not frequency-weighted. The numerical convention for bases
is the same as in the two-channel version.

The NPZ archive contains `z_u16.npy` (an H×W little-endian uint16 array) and
`metadata.json`: S, formula, conventions, dimensions, number of bases,
final cumulative sum, maximum z, source, and verification. ZIP is used without
additional compression. No intermediate JSON is written to disk; intermediate
data is built in RAM for each image. `--verify` checks decoded transformed
chunks, not the accuracy of the representation.

```python
import json
import numpy as np
from zipfile import ZipFile

path = 'output/cumulative_rank_s1/image_00000.npz'
with np.load(path, allow_pickle=False) as data:
    z = data['z_u16']                       # (H, W), uint16
    x = z.astype(np.float32)[None] / 65535  # (1, H, W), only when loading
with ZipFile(path) as archive:
    metadata = json.loads(archive.read('metadata.json'))
```

Python can invoke the same command with `subprocess.run([...], check=True)`.
Batch generation and training are not started automatically. Keep separate
directories for each S and representation.

## V3: adaptive scale per image

V3 retains ranks and deltas `a_r=log2(1+d_r)/log2(1+d_max)` and uses
`q_r=floor(S_i*a_r)`, `z_r=r+sum(q_0..q_r)`. Binary search selects the largest
nonnegative integer S_i with `K-1+sum(q)<=65535`, also accounting for the gap
distribution. V1 and V2 retain their previous behavior (in particular,
V2 uses round, not floor).

One base: S_i=0 and a zero channel. Two bases: S_i=65534. Up to 65536 bases,
S_i=0 is allowed; beyond that count, export fails without clipping.

```powershell
cargo build --manifest-path src/igd_extract/Cargo.toml --bin igd_extract --target-dir src/gdcompress/target --release --offline --locked
python src/batch_network_features.py --version v3 --input-dir data/cifar-10_compressed_1x1 --output-dir output/v3_cumulative_rank_adaptive --verify
```

V3 does not accept `--scale`. To resume, add `--resume`: ZIP integrity,
representation, scale metadata, and requested verification are checked.
The direct command uses `--representation cumulative-rank-adaptive`.

Each NPZ contains `z_u16.npy` and `metadata.json`, with `representation.scale`,
formula, selection method, K (`num_bases`), and `z_max`. Representation name:
`cumulative-rank-u16-v3`. The network can still use `float32(z_u16)/65535`.

Upon batch completion, `scales_v3.csv` collects input, output, S_i, K_i, and
z_max for all command inputs, including those skipped on resume. The CSV is
regenerated atomically; if interrupted, metadata in individual NPZs remains
the source of values, and the previous CSV may be incomplete. Loading the CSV
is not required to use the tensors in the network.
