"""Read existing IGD headers and file sizes; never recompress or decode pixels."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import struct
import time

ROOT = Path(__file__).resolve().parents[2]


def image_id(value: str) -> str:
    return Path(value.replace("\\", "/")).stem.removesuffix(".network")


def read_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(32)
    if len(header) != 32 or header[:4] != b"IGD\x01":
        raise ValueError(f"Invalid or unsupported IGD header: {path}")
    width, height = struct.unpack_from("<II", header, 4)
    channels = header[12]
    group_w, group_h, payload = struct.unpack_from("<IIQ", header, 16)
    if not width or not height or channels not in (3, 4) or not group_w or not group_h:
        raise ValueError(f"Invalid image metadata: {path}")
    if payload != size - 32:
        raise ValueError(f"IGD payload length mismatch: {path}")
    raw = width * height * channels  # Original RGB/RGBA, 8 bits per channel.
    return dict(width=width, height=height, channels=channels,
                group_width=group_w, group_height=group_h,
                colorspace=header[13], color_model=header[14],
                grouping_transform=header[15],
                raw_bytes=raw, igd_bytes=size,
                saving_raw_pct=100 * (1 - size / raw),
                raw_to_igd_ratio=raw / size,
                bits_per_pixel=8 * size / (width * height))


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    values = sorted(values)
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return dict(count=len(values), mean=statistics.fmean(values),
                std_population=statistics.pstdev(values), min=values[0],
                p05=percentile(.05), median=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), max=values[-1])


def summarize(rows: list[dict]) -> dict:
    raw = sum(r["raw_bytes"] for r in rows)
    igd = sum(r["igd_bytes"] for r in rows)
    return dict(count=len(rows), raw_bytes=raw, igd_bytes=igd,
                total_saving_raw_pct=100 * (1 - igd / raw) if raw else None,
                total_raw_to_igd_ratio=raw / igd if igd else None,
                expanded_vs_raw_count=sum(r["igd_bytes"] > r["raw_bytes"] for r in rows),
                metrics={key: describe([r[key] for r in rows if r.get(key) is not None])
                         for key in ("igd_bytes", "saving_raw_pct", "raw_to_igd_ratio",
                                     "bits_per_pixel", "png_bytes", "saving_png_pct", "K", "base_bits")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--igd-root", type=Path, default=ROOT / "data/cifar-10_compressed_1x1")
    parser.add_argument("--original-root", type=Path, default=ROOT / "data/cifar-10_resized")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k-report", type=Path)
    parser.add_argument("--k-report-split", help="Explicit split for legacy reports without split identifiers")
    parser.add_argument("--split-file", type=Path, help="Saved train_ids/val_ids JSON; assigns train rows to training/validation")
    parser.add_argument("--k-min", type=int, default=32, help="Analysis threshold only; does not change compression")
    args = parser.parse_args()
    if args.k_min < 1:
        parser.error("--k-min must be positive")
    if args.k_report and not args.k_report_split:
        parser.error("--k-report requires --k-report-split to avoid ambiguous image IDs")
    if args.output_dir.exists():
        parser.error("Output directory already exists; select a new directory")
    paths = sorted(args.igd_root.rglob("*.igd"))
    if not paths:
        parser.error("No IGD files found")
    k_rows = {}
    if args.k_report:
        with args.k_report.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                key = image_id(record["path"])
                if key in k_rows or "error" in record or record["num_bases"] < 1:
                    raise ValueError(f"Duplicate or invalid K record: {key}")
                k_rows[key] = record
    memberships = {}
    if args.split_file:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        for field, name in (("train_ids", "training"), ("val_ids", "validation")):
            for value in split[field]:
                key = image_id(value)
                if key in memberships:
                    raise ValueError(f"Duplicate split membership: {key}")
                memberships[key] = name
    start = time.perf_counter()
    rows, missing_png, seen = [], [], set()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(args.igd_root)
        split_name = relative.parts[0] if len(relative.parts) > 1 else "unspecified"
        key = (split_name, path.stem)
        if key in seen:
            raise ValueError(f"Ambiguous image identity: {key}")
        seen.add(key)
        row = dict(split=split_name, image_id=path.stem, igd_path=relative.as_posix(),
                   evaluation_partition=None, K=None, base_bits=None,
                   below_k_min=None, png_bytes=None, saving_png_pct=None,
                   compression_seconds=None, **read_header(path))
        if split_name == "train" and memberships:
            if path.stem not in memberships:
                raise ValueError(f"Missing split membership: {path.stem}")
            row["evaluation_partition"] = memberships[path.stem]
        if split_name == args.k_report_split and args.k_report:
            record = k_rows.get(path.stem)
            if record is None:
                raise ValueError(f"Missing K report entry: {path}")
            if (record["width"], record["height"]) != (row["width"], row["height"]):
                raise ValueError(f"K report dimensions differ: {path}")
            row.update(K=record["num_bases"], base_bits=record["base_bits"],
                       below_k_min=record["num_bases"] < args.k_min)
        png = args.original_root / relative.with_suffix(".png")
        if png.is_file():
            row["png_bytes"] = png.stat().st_size
            if row["png_bytes"]:
                row["saving_png_pct"] = 100 * (1 - row["igd_bytes"] / row["png_bytes"])
        else:
            missing_png.append(relative.with_suffix(".png").as_posix())
        rows.append(row)
        if index % 10000 == 0:
            print(f"Read {index}/{len(paths)} IGD headers", flush=True)
    groups = {"all": summarize(rows)}
    for split_name in sorted({r["split"] for r in rows}):
        groups[f"split/{split_name}"] = summarize([r for r in rows if r["split"] == split_name])
    for partition in ("training", "validation"):
        selected = [r for r in rows if r["evaluation_partition"] == partition]
        if selected:
            groups[f"partition/{partition}"] = summarize(selected)
    for group, selected in [("all", rows)] + [
        (p, [r for r in rows if r["evaluation_partition"] == p]) for p in ("training", "validation")]:
        for low in (True, False):
            subset = [r for r in selected if r["below_k_min"] is low]
            if subset:
                groups[f"{group}/K{'<' if low else '>='}{args.k_min}"] = summarize(subset)
    report = dict(schema="igd-compression-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  configuration={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  conventions=dict(raw="width * height * channels bytes, RGB/RGBA uint8 before compression",
                                   saving="100 * (1 - IGD bytes / reference bytes); negative means expansion",
                                   igd="Actual file size including headers and all encoded sections",
                                   validation="Header and payload length only; not a lossless round-trip verification",
                                   k="Existing report trusted; dimensions checked, no dictionary revalidation",
                                   timing="Historical compression time unavailable; audit time is not compression time",
                                   percentiles="Linear interpolation; population standard deviation"),
                  groups=groups, missing_png=missing_png,
                  compression_timing=None, audit_elapsed_seconds=time.perf_counter() - start)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_image.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "all": groups["all"]}, indent=2))


if __name__ == "__main__":
    main()
