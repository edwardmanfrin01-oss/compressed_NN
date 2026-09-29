# gdcompress — Generalized Deduplication Compressor

**gdcompress** is a lossless compression tool based on **Generalized Deduplication (GD)** and the **EntroGD** algorithm. It supports both **tabular data** (CSV) and **images** (PNG, JPEG, BMP, etc.), producing compressed files that preserve bit-exact reconstruction, allow **random access** to individual rows without decompressing the entire file, and enable **analytics directly on compressed data** through condensed samples.

Developed as part of a master's thesis at **Aarhus University** under the supervision of **Daniel Enrique Lucani Rötter**, with a primary focus on image compression. The underlying EntroGD engine equally supports tabular workloads.

## Features

- **Lossless compression** — bit-exact roundtrip for both tabular and image data
- **Random access** — decompress specific rows or regions without decompressing everything
- **Compressed-domain analytics** — extract condensed, representative samples directly from `.tgd`/`.igd`/`.egd` files
- **Pluggable filter pipeline** — compose preprocessing, encoding, and base-selection strategies via a modular chain-of-filters architecture
- **Multiple encoding strategies** — flat encoding, RLE with offsets, and canonical Huffman coding
- **Adaptive base-bit selection** — entropy-driven with HyperLogLog approximate counting for compute-efficient base configuration
- **Polars-based data pipeline** — tabular ingestion, preprocessing, and reconstruction are built on the [Polars](https://pola.rs) DataFrame library, making it straightforward to add support for additional data sources (Parquet, IPC, JSON, etc.)
- **Numerical-only tabular support** — currently handles integer and floating-point columns; string/categorical support is not yet implemented
- **Three container formats** — `.tgd` (tabular) and `.igd` (image) are thin wrappers around the internal `.egd` (EntroGD Generic Data) format, each adding domain-specific metadata (column schema for tabular, image dimensions/color model for images)
- **Parallel processing** — Rayon-powered parallelism where applicable

## Thesis Contributions

Beyond prior published work (EntroGD, RAGE), the following contributions were developed as part of the thesis:

- **Practical implementation** of the full EntroGD pipeline and the lossless compression components of RAGE in a unified, production-quality Rust codebase.
- **Faster base-bit selection** via a modified HyperLogLog-based approximate counting method, replacing precise counting for speed while retaining near-optimal base configurations.
- **Image preprocessing pipeline**: YCoCg-R color space conversion, pixel grouping, and offset-transform coding (`ForFirstPixel`/`ForMin`) with zigzag-encoded intra-group deltas for improved compression of natural images.
- **Huffman encoding** of deviation streams via canonical Huffman coding, improving compression ratios for continuous-tone images such as natural photographs.
- **Base table compression**: entropy-guided sorting of base-table rows followed by delta encoding (`DeltaEncodeBaseTableFixed`), reducing the base-table overhead significantly.

## Installation

**Prerequisites:** [Rust](https://rust-lang.org) (stable toolchain).

```bash
git clone https://github.com/Voggo/gdcompress.git
cd gdcompress
cargo build --release
```

The binary will be at `./target/release/gdcompress`.

> **Note:** The project's `config.toml` sets `-C target-cpu=native` for CPU-specific optimizations. If you plan to distribute the binary, remove or adjust this flag.

## Usage

Run `gdcompress --help` for the full option listing.

```text
gdcompress  [OPTIONS]  <INPUT>
```

The action (compress, decompress, or analytics) is **auto-detected from the input file extension**:

| Input extension            | Action             | Default output extension |
|----------------------------|--------------------|--------------------------|
| `.csv`                     | Compress tabular   | `.tgd`                   |
| `.tgd`                     | Decompress tabular | `.csv`                   |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.webp` | Compress image | `.igd` |
| `.igd`                     | Decompress image   | `.png`                   |
| `.egd`                     | Decompress generic | *(requires `-o`)*  — internal format, use `.tgd`/`.igd` for normal workflows |

### Options

| Flag                     | Default    | Description |
|--------------------------|------------|-------------|
| `-o`, `--output`         | *(auto)*   | Output file path (auto-derived from input if omitted) |
| `--analytics`            | off        | Show condensed samples from a compressed file and write `input.analytics.csv` |
| `-v`, `--verbose`        | off        | Verbose output with detailed statistics |
| `--pixel-grouping`       | `"3x3"`    | Pixel grouping size for image compression (e.g. `"2x2"`, `"3x3"`) |
| `--float-scaling`        | `"on"`     | Float-to-integer conversion for tabular compression (`on` or `off`) |
| `--float-precision`      | `9`        | Decimal places preserved when float scaling is on (tabular only) |
| `--rows`                 | *(all)*    | Decompress only specific rows, e.g. `"0,5,10-20"` (tabular only) |
| `--crop`                 | *(none)*   | Decompress a region of an image: `"x1,y1,x2,y2"` (`.igd` only) |
| `--encoding`             | *(context)*| Encoding strategy: `normal`, `rle`, or `huffman` (default: `normal` for tabular, `huffman` for image) |
| `--base-compression`     | *(context)*| Delta base-table compression: `on` or `off` (default: `off` for tabular, `on` for image) |
| `--base-counting`        | `"precise"`| Base-bit counting strategy: `precise` or `approximate` |

### Examples

Compress a CSV file:

```bash
gdcompress data/tabular/aarhus-citylab.csv
# Produces: data/tabular/aarhus-citylab.tgd
```

Compress an image:

```bash
gdcompress data/images/kodim10.png -o compressed/kodim10.igd
```

Decompress a tabular file with random access (rows 0, 5, and 10–20):

```bash
gdcompress data/tabular/aarhus-citylab.tgd --rows "0,5,10-20" -o subset.csv
```

Decompress an image:

```bash
gdcompress compressed/kodim10.igd -o restored.png
```

Extract condensed analytics samples:

```bash
gdcompress data/tabular/aarhus-citylab.tgd --analytics
# Prints sample summary and writes data/tabular/aarhus-citylab.analytics.csv
```

Decompress a cropped region of an image:

```bash
gdcompress compressed/kodim10.igd --crop "100,50,399,349" -o crop.png
# Decompresses only the 300x300 pixel region from (100,50) to (399,349)
```

Use a different encoding or base-table compression:

```bash
# Tabular with Huffman encoding and base-table delta compression
gdcompress data/tabular/aarhus-citylab.csv --encoding huffman --base-compression on

# Image with RLE encoding and no base-table compression
gdcompress data/images/kodim10.png --encoding rle --base-compression off -o out.igd
```

## Benchmarks

Benchmarks are located in `benches/`. They focus on image compression but exercise the full pipeline including base selection, encoding, delta encoding, and color models.

```bash
# Per-stage micro-benchmarks of each pipeline component
cargo bench --bench benchmark_individual

# Compression ratio impact of individual design choices
cargo bench --bench benchmark_compression_ratio

# Full compression/decompression comparison against PNG, WebP, QOI, and JPEG-LS
cargo bench --bench benchmark
```

Benchmark results are written as CSV files under `target/bench-results/`.

**Filtering benchmarks:** For `benchmark_individual`, set the `BENCH_FILTER` environment variable to narrow which pipeline stages are measured (e.g. `BENCH_FILTER=Entropy`).

## Citing

If you use gdcompress in your research, please cite the underlying work:

**EntroGD** — the core algorithm:

```bibtex
@inproceedings{zhao2026entrogd,
  title     = {{EntroGD}: Scalable Generalized Deduplication for Efficient
               Direct Analytics on Compressed {IoT} Data},
  author    = {Zhao, Xiaobo and Lucani, Daniel E.},
  booktitle = {{IEEE INFOCOM 2026 -- IEEE Conference on Computer Communications}},
  year      = {2026},
  publisher = {IEEE},
}
```

**RAGE** — generalized deduplication applied to image compression:

```bibtex
@inproceedings{rask2024rage,
  title     = {{RAGE} for the Machine: Image Compression with Low-Cost
               Random Access for Embedded Applications},
  author    = {Rask, Christian D. and Lucani, Daniel E.},
  booktitle = {2024 IEEE International Conference on Image Processing (ICIP)},
  year      = {2024},
  publisher = {IEEE},
}
```

## License

MIT

## Author

**Niels Viggo Stark Madsen**

This project was developed as part of a master's thesis at **Aarhus University**, supervised by **Daniel Enrique Lucani Rötter**.
