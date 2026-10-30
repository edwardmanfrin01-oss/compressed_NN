"""Compress PNG splits, optionally reusing baseline IGDs that meet --k-min."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress images with gdcompress without mixing grouping configurations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_resized/test",
        help="Directory containing PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination (default: data/cifar-10_compressed_1x1_Kmin/<train or test>).",
    )
    parser.add_argument(
        "--pixel-grouping",
        default="1x1",
        help="Grouping passed to gdcompress, for example 1x1 or 3x3 (default: 1x1).",
    )
    parser.add_argument(
        "--gdcompress-bin",
        type=Path,
        default=PROJECT_ROOT / "src" / "gdcompress" / "target" / "release" / "gdcompress.exe",
        help="Path to the gdcompress executable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IGD files that are already present in the output directory.",
    )
    parser.add_argument("--k-min", type=int, help="Optional minimum number of distinct bases.")
    parser.add_argument("--existing-dir", type=Path,
                        help="Baseline IGD split directory. With --k-min, defaults to data/cifar-10_compressed_1x1/<input split>.")
    parser.add_argument("--base-counts-bin", type=Path,
                        default=PROJECT_ROOT / "src/gdcompress/target/release/igd_base_counts.exe")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "data/cifar-10_compressed_1x1_Kmin"
        if args.input_dir.name in ("train", "test"):
            args.output_dir /= args.input_dir.name
    if args.k_min is not None and args.k_min < 1:
        parser.error("--k-min must be positive")
    if args.existing_dir and args.k_min is None:
        parser.error("--existing-dir requires --k-min")
    if args.k_min is not None and args.existing_dir is None:
        if args.input_dir.name not in ("train", "test"):
            parser.error("Use --existing-dir for input directories other than train/test")
        args.existing_dir = PROJECT_ROOT / "data/cifar-10_compressed_1x1" / args.input_dir.name
    return args


def read_base_count(process, path: Path, grouping: str) -> int:
    """Query one IGD in a persistent helper, avoiding one process per image."""
    process.stdin.write(str(path.resolve()) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"IGD metadata helper failed for {path}; see its stderr")
    k, width, height, group_w, group_h = map(int, response.split())
    if f"{group_w}x{group_h}" != grouping:
        raise ValueError(f"Baseline grouping differs from requested grouping: {path}")
    return k


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.gdcompress_bin.is_file():
        raise SystemExit(
            f"gdcompress executable not found: {args.gdcompress_bin}. "
            "Build it first with cargo build --release."
        )
    if args.k_min is not None:
        if not args.existing_dir.is_dir() or not args.base_counts_bin.is_file():
            raise SystemExit("Baseline directory or igd_base_counts executable missing. Build with cargo build --release --bins.")
        if args.output_dir.resolve() == args.existing_dir.resolve():
            raise SystemExit("Output must differ from the baseline directory.")

    image_files = sorted(args.input_dir.glob("*.png"))
    if not image_files:
        raise SystemExit(f"No PNG files found in: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_files)} images; pixel grouping: {args.pixel_grouping}")
    print(f"Output directory: {args.output_dir}")
    helper = None
    if args.k_min is not None:
        helper = subprocess.Popen([str(args.base_counts_bin)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8")

    try:
        process_images(args, image_files, helper)
    finally:
        if helper is not None:
            helper.stdin.close()
            helper.stdout.close()
            helper.wait()


def process_images(args, image_files, helper):
    compressed = skipped = copied = 0
    # A new log for each invocation also records an interrupted run's completed images.
    log_path = args.output_dir / f"compression_actions.csv"
    with log_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "action", "original_K", "achieved_K", "requested_K", "threshold_reached"])
        for index, image_path in enumerate(image_files, 1):
            output_path = args.output_dir / f"{image_path.stem}.igd"
            if output_path.exists():
                if args.resume:
                    skipped += 1
                    continue
                raise SystemExit(f"Output already exists: {output_path}. Use --resume to skip existing files.")
            original_k = None
            if helper is not None:
                baseline = args.existing_dir / output_path.name
                if baseline.is_file():
                    original_k = read_base_count(helper, baseline, args.pixel_grouping)
            if original_k is not None and original_k >= args.k_min:
                shutil.copy2(baseline, output_path)
                copied += 1
                achieved = original_k
                action = "copied"
            else:
                command = [str(args.gdcompress_bin), str(image_path), "--pixel-grouping",
                           args.pixel_grouping, "-o", str(output_path)]
                if args.k_min is not None:
                    command.extend(["--k-min", str(args.k_min)])
                try:
                    # Hide per-image progress; retain diagnostics if compression fails.
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
                except subprocess.CalledProcessError as error:
                    raise SystemExit(f"Compression failed for {image_path.name}: {error}\n{error.stderr}") from error
                compressed += 1
                achieved = read_base_count(helper, output_path, args.pixel_grouping) if helper else None
                action = "recompressed" if original_k is not None else "compressed"
            writer.writerow([image_path.name, action, original_k, achieved, args.k_min,
                             achieved >= args.k_min if achieved is not None else ""])
            stream.flush()
            if index % 500 == 0 or index == len(image_files):
                print(f"Processed {index}/{len(image_files)} images", flush=True)
    print(f"Compressed: {compressed}; copied: {copied}; skipped: {skipped}")
    print(f"Action report: {log_path}")


if __name__ == "__main__":
    main()
