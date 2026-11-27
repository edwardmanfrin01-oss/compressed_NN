"""Read existing IGD headers and file sizes; never recompress or decode pixels."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import struct
import time
from zipfile import ZipFile

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


def index_npz(root: Path) -> dict:
    """Match archives by split and image ID, rejecting ambiguous aliases."""
    if not root.is_dir():
        raise ValueError(f"NPZ root is not a directory: {root}")
    result = {}
    for path in sorted(root.rglob("*.npz")):
        relative = path.relative_to(root)
        split = relative.parts[0] if len(relative.parts) > 1 else "unspecified"
        key = (split, image_id(path.name))
        if key in result:
            raise ValueError(f"Ambiguous NPZ identity {key}: {result[key]} and {path}")
        result[key] = path
    return result


def read_npz_sizes(path: Path, width: int, height: int) -> dict:
    """Read NPY headers only; report stored tensor bytes without loading pixels."""
    arrays = []
    tensor_bytes = 0
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP entries: {path}")
        array_names = {name for name in names if name.endswith(".npy")}
        if array_names not in ({"z_u16.npy"}, {"rank_u8.npy", "delta_u8.npy"}):
            raise ValueError(f"Unsupported NPZ tensor layout: {path}")
        for name in sorted(array_names):
            with archive.open(name) as stream:
                magic = stream.read(8)
                if magic[:6] != b"\x93NUMPY" or magic[6:] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                    raise ValueError(f"Invalid NPY version: {path}:{name}")
                length_size = 2 if magic[6] == 1 else 4
                length_raw = stream.read(length_size)
                if len(length_raw) != length_size:
                    raise ValueError(f"Truncated NPY header: {path}:{name}")
                length = int.from_bytes(length_raw, "little")
                if length > 65536:
                    raise ValueError(f"NPY header exceeds audit limit: {path}:{name}")
                header = stream.read(length)
                metadata = ast.literal_eval(header.decode("utf-8" if magic[6] == 3 else "latin1"))
                expected = ("<u2", "=u2") if name == "z_u16.npy" else ("|u1", "<u1", "=u1")
                if (metadata.get("descr") not in expected
                        or metadata.get("shape") != (height, width)
                        or type(metadata.get("fortran_order")) is not bool):
                    raise ValueError(f"Unexpected NPY shape/dtype: {path}:{name}")
                size = math.prod(metadata["shape"]) * (2 if name == "z_u16.npy" else 1)
                if len(header) != length or archive.getinfo(name).file_size != 8 + length_size + length + size:
                    raise ValueError(f"NPY payload length mismatch: {path}:{name}")
                tensor_bytes += size
                arrays.append(dict(name=name, shape=metadata["shape"], dtype=metadata["descr"], bytes=size))
    return dict(npz_bytes=path.stat().st_size, tensor_bytes=tensor_bytes, npz_arrays=arrays)


def summarize(rows: list[dict]) -> dict:
    raw = sum(r["raw_bytes"] for r in rows)
    igd = sum(r["igd_bytes"] for r in rows)
    paired = [r for r in rows if r.get("png_bytes", 0)]
    png = sum(r["png_bytes"] for r in paired)
    paired_igd = sum(r["igd_bytes"] for r in paired)
    result = dict(count=len(rows), raw_bytes=raw, igd_bytes=igd,
                total_saving_raw_pct=100 * (1 - igd / raw) if raw else None,
                total_raw_to_igd_ratio=raw / igd if igd else None,
                expanded_vs_raw_count=sum(r["igd_bytes"] > r["raw_bytes"] for r in rows),
                png_comparison_count=len(paired), png_bytes=png,
                paired_igd_bytes=paired_igd,
                total_saving_png_pct=100 * (1 - paired_igd / png) if png else None,
                total_png_to_igd_ratio=png / paired_igd if paired_igd else None,
                expanded_vs_png_count=sum(r["igd_bytes"] > r["png_bytes"] for r in paired),
                metrics={key: describe([r[key] for r in rows if r.get(key) is not None])
                         for key in ("igd_bytes", "saving_raw_pct", "raw_to_igd_ratio",
                                     "bits_per_pixel", "png_bytes", "saving_png_pct", "K", "base_bits",
                                     "npz_bytes", "tensor_bytes", "npz_reduction_raw_pct")
                         if any(r.get(key) is not None for r in rows)})
    if any("npz_bytes" in r for r in rows):
        matched = [r for r in rows if r.get("npz_bytes") is not None]
        matched_raw = sum(r["raw_bytes"] for r in matched)
        npz = sum(r["npz_bytes"] for r in matched)
        result.update(npz_comparison_count=len(matched), npz_bytes=npz,
                      tensor_bytes=sum(r["tensor_bytes"] for r in matched),
                      npz_paired_raw_bytes=matched_raw,
                      total_npz_reduction_raw_pct=100 * (1 - npz / matched_raw) if matched_raw else None)
    return result


def write_readable_report(report: dict, path: Path) -> None:
    lines = ["# Existing compression statistics", "",
             "IGDs compared with raw RGB/RGBA at 8 bits per channel and existing PNGs.",
             "Savings = 100 * (1 - IGD bytes / reference bytes). Negative values indicate expansion.",
             "Aggregate savings use byte totals.", "",
             "| Group | Images | Aggregate RGB savings | Aggregate PNG savings | Expansions RGB | Expansions PNG |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, group in report["groups"].items():
        png = group["total_saving_png_pct"]
        png_text = f"{png:.2f}%" if png is not None else "N/A"
        raw = group["total_saving_raw_pct"]
        raw_text = f"{raw:.2f}%" if raw is not None else "N/A"
        lines.append(f"| {name} | {group['count']} | {raw_text} | {png_text} | "
                     f"{group['expanded_vs_raw_count']} | {group['expanded_vs_png_count']} |")
    for name, group in report["groups"].items():
        lines.extend(["", f"## {name}", "",
                      "| Savings (%) | Mean | Population std | Min | P05 | Meann | P95 | P99 | Max |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for metric, label in (("saving_raw_pct", "Raw RGB"), ("saving_png_pct", "PNG")):
            stats = group["metrics"].get(metric)
            if stats:
                values = " | ".join(f"{stats[k]:.2f}" for k in
                                    ("mean", "std_population", "min", "p05", "median", "p95", "p99", "max"))
                lines.append(f"| {label} | {values} |")
        lines.extend(["", f"Raw bytes: {group['raw_bytes']:,}; IGD bytes: {group['igd_bytes']:,}.",
                      f"Matched, nonempty PNGs: {group['png_comparison_count']:,}."])
    lines.extend(["", f"Missing PNGs: {len(report['missing_png'])}.",
                  "Checks: IGD header and payload length; no reconstruction verification.",
                  "No recompression or compression timing measurements.", ""])
    if "missing_npz" in report:
        lines.extend(["## NPZ feature storage", "",
                      "Feature sizes describe a classification representation, not lossless RGB compression.",
                      "Tensor bytes use the stored dtype and exclude archive overhead; float32 loading is not measured.",
                      "Matching checks split, image ID, shape, and dtype, not content equivalence or payload CRC.", "",
                      "| Group | Matched NPZs | NPZ bytes | Tensor bytes | Reduction vs matched raw RGB/RGBA |",
                      "|---|---:|---:|---:|---:|"])
        for name, group in report["groups"].items():
            reduction = group["total_npz_reduction_raw_pct"]
            text = f"{reduction:.2f}%" if reduction is not None else "N/A"
            lines.append(f"| {name} | {group['npz_comparison_count']} | {group['npz_bytes']:,} | "
                         f"{group['tensor_bytes']:,} | {text} |")
        for name, group in report["groups"].items():
            lines.extend(["", f"### {name} NPZ size distribution", "",
                          "| Bytes | Mean | Population std | Min | P05 | Median | P95 | P99 | Max |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for key in ("npz_bytes", "tensor_bytes"):
                stats = group["metrics"].get(key)
                if stats:
                    values = " | ".join(f"{stats[k]:.2f}" for k in
                                        ("mean", "std_population", "min", "p05", "median", "p95", "p99", "max"))
                    lines.append(f"| {key} | {values} |")
        lines.extend(["", f"Missing NPZs: {len(report['missing_npz'])}.",
                      f"NPZs without a matching IGD: {len(report['unmatched_npz'])}.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--igd-root", type=Path, default=ROOT / "data/cifar-10_compressed_1x1")
    parser.add_argument("--original-root", type=Path, default=ROOT / "data/cifar-10_resized")
    parser.add_argument("--npz-root", type=Path, help="Optional feature archives with the same train/test layout as --igd-root")
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
    npz_index = index_npz(args.npz_root) if args.npz_root is not None else {}
    missing_npz, used_npz = [], set()
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
                   **read_header(path))
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
        if args.npz_root is not None:
            row.update(npz_path=None, npz_bytes=None, tensor_bytes=None,
                       npz_arrays=None, npz_reduction_raw_pct=None)
            npz = npz_index.get(key)
            if npz is None:
                missing_npz.append(relative.with_suffix(".network.npz").as_posix())
            else:
                used_npz.add(key)
                sizes = read_npz_sizes(npz, row["width"], row["height"])
                arrays = sizes.pop("npz_arrays")
                row.update(sizes, npz_path=npz.relative_to(args.npz_root).as_posix(),
                           npz_arrays=json.dumps(arrays),
                           npz_reduction_raw_pct=100 * (1 - sizes["npz_bytes"] / row["raw_bytes"]))
        rows.append(row)
        if index % 10000 == 0:
            print(f"Read {index}/{len(paths)} IGD headers", flush=True)
    if not args.k_report:
        for row in rows:
            for field in ("K", "base_bits", "below_k_min"):
                row.pop(field)
    if not args.split_file:
        for row in rows:
            row.pop("evaluation_partition")
    groups = {"all": summarize(rows)}
    for split_name in sorted({r["split"] for r in rows}):
        groups[f"split/{split_name}"] = summarize([r for r in rows if r["split"] == split_name])
    for partition in ("training", "validation"):
        selected = [r for r in rows if r.get("evaluation_partition") == partition]
        if selected:
            groups[f"partition/{partition}"] = summarize(selected)
    for group, selected in [("all", rows)] + [
        (p, [r for r in rows if r.get("evaluation_partition") == p]) for p in ("training", "validation")]:
        for low in (True, False):
            subset = [r for r in selected if r.get("below_k_min") is low]
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
    if args.npz_root is not None:
        report.update(missing_npz=missing_npz,
                      unmatched_npz=[npz_index[key].relative_to(args.npz_root).as_posix()
                                     for key in sorted(npz_index.keys() - used_npz)])
        report["conventions"].update(
            npz="Archive file bytes include metadata and ZIP/NPY overhead; tensor_bytes excludes overhead and uses the stored dtype, not float32",
            npz_comparison="Same split/image ID and matching dimensions; content/provenance and ZIP payload CRC are not verified. Features are not a lossless RGB replacement")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_image.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_readable_report(report, args.output_dir / "report.md")
    print(json.dumps({"output": str(args.output_dir), "all": groups["all"]}, indent=2))


if __name__ == "__main__":
    main()
