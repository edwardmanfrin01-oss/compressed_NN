# Baseline without geometry: dictionary deltas

## Objective

Use a minimal experiment to check whether deltas between sorted bases contain
information useful for classifying CIFAR-10 images. This baseline uses only
deltas. It excludes spatial IDs, frequencies, minimum bases, deviations, and
masks. It is not a lossless image representation and does not use ResNet50.

## Procedure performed on September 28, 2026

1. Random balanced selection of 40 PNGs per class from training files already
   resized to 224x224. Labels: `data/cifar-10_labels.json`. Seed: 42.
2. Per-image, per-class split: 32 train + 8 validation. Totals: 320 train and
   80 validation. This is not the official CIFAR-10 test set.
3. Temporary compression of each PNG with `--pixel-grouping 1x1`, using the
   existing compressor. The compressor's color transformations remain enabled.
4. Reading with the existing Rust extractor; the generic JSON is temporary only.
5. Interpreting each base string as an MSB-first integer, using the same convention
   as the previous experiment. This convention is not an RGB intensity and does
   not describe the entropy-based selection order.
6. Sorting bases in ascending order and subtracting consecutive values exactly:
   K bases produce K-1 deltas. Deltas are NOT sorted again after subtraction.
7. Numerical conversion `log2(1 + delta) / 24`. The denominator is fixed, not
   estimated from the dataset. Exact deltas remain in the file as `uint32`.
8. Dividing the sequence axis into 256 equal intervals and averaging within each
   interval (with fractional weights at the boundaries). Short sequences have
   their values distributed across multiple intervals, without zero padding.
   A single base conventionally produces a zero vector because no differences
   between bases exist.
9. Standardizing the 256 features using mean/std from the training set ONLY,
   clipping to [-10,10], MLP 256 -> 64 ReLU -> 10 softmax. Adam, learning
   rate 0.001, L2 on weights 0.001, batch 32, 30 epochs, seed 42.

Averaging reduces sequence detail and does not preserve the number of bases.
Dictionary order is preserved, but the full original sequence is not.
Image geometry is absent: images with the same dictionary produce the same
input even if their pixels have different spatial arrangements.

## Pilot result

| Quantity | Result |
|---|---:|
| Training / validation images | 320 / 80 |
| Model parameters | 17,098 |
| Epochs | 30 |
| Final training accuracy | 82.8125% |
| Final and best validation accuracy | 18.75% (15/80) |
| Uniform random baseline | 10% |
| Deltas per image, min / median / max | 1 / 887.5 / 18,401 |
| Selected bits per base, min / max | 1 / 24 |
| NPZ dataset, including exact deltas and metadata | 679,850 bytes |
| Model input per image | 256 float32 = 1,024 bytes |

The large train/validation gap indicates overfitting. With 80 validation images,
we cannot conclude that the representation generalizes well. The best epoch
is selected on validation; this is neither an independent test measurement
nor a controlled comparison with ResNet50.

Preparation took about 170 seconds on this computer. The network is very small
and training took less than one second (excluding startup/imports).
These are indicative timings, not a benchmark.

## Files and commands

- Code: `src/delta_baseline.py` (NumPy; PyTorch is not required).
- Tests: `src/test_delta_baseline.py`.
- Generated data: `output/delta_baseline/pilot.npz`.
- Metrics, configuration, and split lists: `runs/delta_baseline/pilot/report.json`.
- Model with the best validation result: `runs/delta_baseline/pilot/best_model.npz`.

Generated files already exist; use new names to repeat the experiment.
From the AARHUS root, with Python + NumPy available:

```powershell
python src/delta_baseline.py prepare --per-class 40 --output output/delta_baseline/pilot_v2.npz
python src/delta_baseline.py train --data output/delta_baseline/pilot_v2.npz --run-dir runs/delta_baseline/pilot_v2
python -m unittest discover -s src -p test_delta_baseline.py
```

In the terminal used here, `python` is not on PATH. To use the runtime that
was used for this experiment:

```powershell
$deltaPython = 'C:/Users/edward/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $deltaPython src/delta_baseline.py --help
```

Replace `python` in the commands above with `& $deltaPython`.

The dataset saves `X` (N,256), `y`, `filenames`, `validation`, `base_bits`,
`delta_values`, and `delta_offsets`, along with metadata. For image i, the
original deltas are `delta_values[delta_offsets[i]:delta_offsets[i+1]]`.
No masks or padding are needed to preserve them. Use
`np.load(path, allow_pickle=False)`. The file also saves repository commits
for context; the hash alone does not represent uncommitted local changes.

## Sizes: correct comparisons

The 256-float32 vector is a lossy summary, not a new lossless compression.
A raw RGB 224x224 image occupies 150,528 bytes; the original 32x32 CIFAR image
occupies 3,072. The vector occupies 1,024 bytes, excluding label/metadata.
The exact sequence instead requires `4*(K-1)` bytes as uint32 and varies by
image. NPZ also uses ZIP compression, so file size must be measured.

To decide whether to continue: increase the sample while keeping the split
fixed, compare multiple seeds, and compare against a control (for example,
only the number of bases or a delta histogram). The saved exact deltas allow
changing the summary without recompressing the images in this sample.
