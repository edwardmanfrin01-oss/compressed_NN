"""Non-spatial delta-only baseline. Only NumPy is required outside the Rust codecs.

Run `prepare` then `train`; use --help on either command. No PyTorch required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "output/delta_baseline/pilot.npz"


def summarize_deltas(deltas: np.ndarray, length: int = 256) -> np.ndarray:
    """Area-average consecutive dictionary intervals into a fixed number of bins.

    Each normalized delta occupies an interval of width one. Fractional bin
    boundaries are integrated exactly for this piecewise-constant signal.
    Short sequences are spread over bins, without padding or an input mask.
    """
    if length < 1:
        raise ValueError("length must be positive")
    if len(deltas) == 0:
        # A single dictionary base has no inter-base gaps.
        return np.zeros(length, dtype=np.float32)
    if np.any(deltas < 0) or np.any(deltas > 2**24 - 1):
        raise ValueError("This baseline supports at most 24 selected base bits")
    values = np.log2(1.0 + deltas.astype(np.float64)) / 24.0
    integral = np.concatenate(([0.0], np.cumsum(values)))
    edges = np.linspace(0.0, len(values), length + 1)
    integrated_edges = np.interp(edges, np.arange(len(values) + 1), integral)
    return (np.diff(integrated_edges) / (len(values) / length)).astype(np.float32)


def deltas_from_document(document: dict) -> tuple[np.ndarray, int]:
    info = document["image"]
    if (info["pixel_group_width"], info["pixel_group_height"]) != (1, 1):
        raise ValueError("Use IGD compressed with --pixel-grouping 1x1")
    width = int(document["base_bits"])
    if not 0 <= width <= 24 or info["channels"] != 3:
        raise ValueError("Expected three channels and at most 24 selected bits")
    # Same convention as igd-network-features-v1: first character is MSB.
    # This is an experimental ordering, not RGB intensity or codec selection order.
    values = sorted(int(base["bits"], 2) if base["bits"] else 0
                    for base in document["bases"])
    if not values or len(set(values)) != len(values):
        raise ValueError("Empty or duplicate base dictionary")
    if any(len(base["bits"]) != width for base in document["bases"]):
        raise ValueError("Inconsistent base lengths")
    # K bases -> K-1 real deltas. Do not insert a fictitious leading zero.
    deltas = np.array([b - a for a, b in zip(values, values[1:])], dtype=np.uint32)
    return deltas, width


def run_codec(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}\n{result.stdout}")


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Choose a new output name: {args.output}")
    if args.per_class < 5 or not 0 < args.validation_fraction < 1:
        raise ValueError("Use at least 5 images per class and a validation fraction in (0, 1)")
    for binary in [args.compressor, args.extractor]:
        if not binary.is_file():
            raise FileNotFoundError(binary)
    rows = json.loads(args.labels.read_text(encoding="utf-8"))
    filenames = [row["file"] for row in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate filenames in label manifest")
    rng = np.random.default_rng(args.seed)
    selected = []
    # Split BEFORE extraction, independently within each class. Only training-set
    # PNGs are used; this held-out subset is validation, not official CIFAR test.
    for label in range(10):
        candidates = [row for row in rows if row["label"] == label]
        if len(candidates) < args.per_class:
            raise ValueError(f"Not enough samples for class {label}")
        chosen = rng.choice(len(candidates), args.per_class, replace=False)
        nval = max(1, min(args.per_class - 1, round(args.per_class * args.validation_fraction)))
        selected.extend((candidates[index], j < nval) for j, index in enumerate(chosen))
    for row, _ in selected:
        if not (args.images / row["file"]).is_file():
            raise FileNotFoundError(args.images / row["file"])

    features, labels, ids, validation, widths, raw, offsets = [], [], [], [], [], [], [0]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aarhus-deltas-") as folder:
        folder = Path(folder)
        for i, (row, is_validation) in enumerate(selected):
            # One image at a time. Temporary IGD/JSON are removed after extraction.
            igd, decoded = folder / "sample.igd", folder / "sample.json"
            run_codec([str(args.compressor), str(args.images / row["file"]),
                       "--pixel-grouping", "1x1", "-o", str(igd)])
            run_codec([str(args.extractor), str(igd), "-o", str(decoded)])
            document = json.loads(decoded.read_text(encoding="utf-8"))
            delta, width = deltas_from_document(document)
            features.append(summarize_deltas(delta, args.length))
            raw.append(delta)
            offsets.append(offsets[-1] + len(delta))
            labels.append(row["label"])
            ids.append(row["file"])
            validation.append(is_validation)
            widths.append(width)
            igd.unlink()
            decoded.unlink()
            if (i + 1) % 25 == 0 or i + 1 == len(selected):
                print(f"Prepared {i + 1}/{len(selected)} in {time.perf_counter() - started:.1f}s", flush=True)

    metadata = dict(schema="delta-only-baseline-v1", seed=args.seed,
                    length=args.length, pixel_grouping="1x1", per_class=args.per_class,
                    normalization="log2(1+delta)/24; consecutive area-average bins",
                    ordering="ascending integer from base bit string interpreted MSB-first",
                    retained="dictionary gaps only; no spatial IDs, deviations, frequencies or min base",
                    split="stratified holdout from CIFAR training set; not official test set",
                    images=str(args.images), labels=str(args.labels),
                    preparation_seconds=time.perf_counter() - started)
    for name, directory in [("project_commit", ROOT), ("compressor_commit", ROOT / "src/gdcompress")]:
        result = subprocess.run(["git", "-c", f"safe.directory={directory.as_posix()}",
                                 "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
        metadata[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    metadata["note"] = "Commits identify repositories; working changes may exist. Save source with experiment."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, X=np.asarray(features), y=np.array(labels, dtype=np.int64),
                            filenames=np.array(ids), validation=np.array(validation),
                            base_bits=np.array(widths, dtype=np.uint8),
                            delta_values=np.concatenate(raw), delta_offsets=np.array(offsets, dtype=np.int64),
                            metadata=np.array(json.dumps(metadata)))
    print(f"Saved {args.output}: {args.output.stat().st_size:,} bytes", flush=True)


def forward(x, w1, b1, w2, b2):
    hidden = np.maximum(x @ w1 + b1, 0)
    logits = hidden @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return hidden, probabilities


def train(args: argparse.Namespace) -> None:
    if args.run_dir.exists():
        raise FileExistsError(f"Choose a new run directory: {args.run_dir}")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    with np.load(args.data, allow_pickle=False) as dataset:
        x, y = dataset["X"].copy(), dataset["y"].copy()
        val = dataset["validation"].copy()
        names = dataset["filenames"].copy()
        metadata = json.loads(str(dataset["metadata"]))
    if not val.any() or val.all() or not np.isfinite(x).all():
        raise ValueError("Invalid dataset or split")
    rng = np.random.default_rng(args.seed)
    # Standardization is fitted ONLY on the training split.
    mean, scale = x[~val].mean(axis=0), x[~val].std(axis=0)
    scale = np.maximum(scale, 1e-5)
    x = np.clip((x - mean) / scale, -10, 10)
    params = [rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], 64)).astype(np.float32),
              np.zeros(64, np.float32),
              rng.normal(0, np.sqrt(2 / 64), (64, 10)).astype(np.float32), np.zeros(10, np.float32)]
    momentum = [np.zeros_like(p) for p in params]
    variance = [np.zeros_like(p) for p in params]
    indices = np.flatnonzero(~val)
    history, step, best = [], 0, -1.0
    args.run_dir.mkdir(parents=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            inputs, targets = x[batch], y[batch]
            hidden, probabilities = forward(inputs, *params)
            grad_logits = probabilities.copy()
            grad_logits[np.arange(len(batch)), targets] -= 1
            grad_logits /= len(batch)
            dw2 = hidden.T @ grad_logits + args.weight_decay * params[2]
            db2 = grad_logits.sum(axis=0)
            grad_hidden = (grad_logits @ params[2].T) * (hidden > 0)
            gradients = [inputs.T @ grad_hidden + args.weight_decay * params[0],
                         grad_hidden.sum(axis=0), dw2, db2]
            step += 1
            for p, grad, m, v in zip(params, gradients, momentum, variance):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad * grad
                p -= args.learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        _, probability = forward(x, *params)
        prediction = probability.argmax(axis=1)
        loss = -np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12))
        row = dict(epoch=epoch, train_loss=float(loss[~val].mean()), val_loss=float(loss[val].mean()),
                   train_accuracy=float((prediction[~val] == y[~val]).mean()),
                   val_accuracy=float((prediction[val] == y[val]).mean()))
        history.append(row)
        if row["val_accuracy"] > best:
            best = row["val_accuracy"]
            np.savez_compressed(args.run_dir / "best_model.npz", w1=params[0], b1=params[1],
                                w2=params[2], b2=params[3], mean=mean, scale=scale)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}: train {row['train_accuracy']:.1%}, val {row['val_accuracy']:.1%}", flush=True)
    report = dict(model="MLP: 256 (or requested length) -> 64 ReLU -> 10 softmax; Adam",
                  data=str(args.data), metadata=metadata, epochs=args.epochs,
                  seed=args.seed, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                  batch_size=args.batch_size, num_parameters=sum(p.size for p in params),
                  train_samples=int((~val).sum()), validation_samples=int(val.sum()),
                  chance_accuracy=0.1, best_validation_accuracy=best,
                  caveat="Exploratory validation on a small held-out subset; not official test accuracy.",
                  final_epoch=history[-1], history=history,
                  train_files=names[~val].tolist(), validation_files=names[val].tolist(),
                  training_seconds=time.perf_counter() - started)
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.run_dir / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="PNG -> temporary 1x1 IGD -> compact delta dataset")
    p.add_argument("--images", type=Path, default=ROOT / "data/cifar-10_resized")
    p.add_argument("--labels", type=Path, default=ROOT / "data/cifar-10_labels.json")
    p.add_argument("--output", type=Path, default=DEFAULT_DATA)
    p.add_argument("--compressor", type=Path, default=ROOT / "src/gdcompress/target/release/gdcompress.exe")
    p.add_argument("--extractor", type=Path, default=ROOT / "src/gdcompress/target/release/igd_extract.exe")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.set_defaults(func=prepare)
    t = commands.add_parser("train", help="Train a small NumPy MLP on delta summaries")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("--run-dir", type=Path, default=ROOT / "runs/delta_baseline/pilot")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--learning-rate", type=float, default=0.001)
    t.add_argument("--weight-decay", type=float, default=0.001)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
