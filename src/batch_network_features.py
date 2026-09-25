"""Export direct IGD-to-network JSON features one image at a time."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sorted-rank and delta feature JSON files directly from .igd files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_compressed_pg_1x1",
        help="Directory containing 1x1 IGD files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "network_features_pg_1x1",
        help="Directory for igd-network-features-v1 JSON files.",
    )
    parser.add_argument(
        "--extractor-bin",
        type=Path,
        default=PROJECT_ROOT / "src" / "gdcompress" / "target" / "release" / "igd_extract.exe",
        help="Path to the igd_extract executable.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify every reconstructed transformed chunk; use first on a small sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip JSON files already present in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.extractor_bin.is_file():
        raise SystemExit(
            f"Extractor executable not found: {args.extractor_bin}. "
            "Build it first with cargo build --release."
        )
    input_files = sorted(args.input_dir.glob("*.igd"))
    if not input_files:
        raise SystemExit(f"No IGD files found in: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_files)} IGD files")
    print(f"Output directory: {args.output_dir}")
    start_time = time.monotonic()
    exported = 0
    skipped = 0
    for index, input_path in enumerate(input_files, 1):
        output_path = args.output_dir / f"{input_path.stem}.network.json"
        if output_path.exists():
            if args.resume:
                skipped += 1
                continue
            raise SystemExit(
                f"Output already exists: {output_path}. Use --resume to skip existing files."
            )
        command = [
            str(args.extractor_bin),
            str(input_path),
            "--network-features",
            "-o",
            str(output_path),
        ]
        if args.verify:
            command.append("--verify")
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Feature export failed for {input_path.name}: {error}") from error
        exported += 1
        if index % 500 == 0 or index == len(input_files):
            print(f"Processed {index}/{len(input_files)} files")

    elapsed = time.monotonic() - start_time
    print(f"Exported: {exported}; skipped: {skipped}; elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
