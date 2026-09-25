"""Compress images one at a time with a reproducible pixel grouping."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress images with gdcompress without mixing grouping configurations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_resized",
        help="Directory containing PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_compressed_pg_1x1",
        help="Directory for IGD files; the default is dedicated to the 1x1 experiment.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.gdcompress_bin.is_file():
        raise SystemExit(
            f"gdcompress executable not found: {args.gdcompress_bin}. "
            "Build it first with cargo build --release."
        )

    image_files = sorted(args.input_dir.glob("*.png"))
    if not image_files:
        raise SystemExit(f"No PNG files found in: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_files)} images; pixel grouping: {args.pixel_grouping}")
    print(f"Output directory: {args.output_dir}")
    start_time = time.monotonic()
    compressed = 0
    skipped = 0

    for index, image_path in enumerate(image_files, 1):
        output_path = args.output_dir / f"{image_path.stem}.igd"
        if output_path.exists():
            if args.resume:
                skipped += 1
                continue
            raise SystemExit(
                f"Output already exists: {output_path}. Use --resume to skip existing files."
            )
        command = [
            str(args.gdcompress_bin),
            str(image_path),
            "--pixel-grouping",
            args.pixel_grouping,
            "-o",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Compression failed for {image_path.name}: {error}") from error
        compressed += 1
        if index % 500 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)} images")

    elapsed = time.monotonic() - start_time
    print(f"Compressed: {compressed}; skipped: {skipped}; elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
