"""
Export IGD features: v1 two uint8 channels, v2 cumulative-rank uint16.

USAGE:
    python src/batch_network_features.py [--version v1|v2] [--scale S] [--input-dir DIR] [--output-dir DIR] [--extractor-bin PATH] [--verify] [--resume]
"""

# from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile, BadZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export v1 rank/delta or v2 cumulative-rank NPZ files from IGD."
    )
    parser.add_argument("--version", choices=("v1", "v2"), default="v1",
                        help="Representation version (default: v1, preserving previous behavior).")
    parser.add_argument("--scale", type=int,
                        help="S for v2, integer 0..65535 (default: 1). Not valid for v1.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_compressed_1x1",
        help="Directory containing 1x1 IGD files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: output/network_features_1x1_v1 or output/v2_cumulative_rank_sX for v2. (X is the 'scale value' selected)",
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
        help="Skip existing NPZs after checking integrity, representation and scale.",
    )
    args = parser.parse_args()
    if args.scale is not None and args.version != "v2":
        parser.error("--scale is only valid with --version v2")
    if args.scale is not None and not 0 <= args.scale <= 65535:
        parser.error("--scale must be in 0..65535")
    if args.version == "v2" and args.scale is None:
        args.scale = 1
    if args.output_dir is None:
        name = "v1_network_features_1x1" if args.version == "v1" else f"v2_cumulative_rank_s{args.scale}"
        args.output_dir = PROJECT_ROOT / "output" / name
    return args


def validate_existing(path: Path, args: argparse.Namespace) -> None:
    """Never silently resume a different representation or a damaged archive."""
    try:
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
            metadata = json.loads(archive.read("metadata.json"))
            expected = ("sorted-base-rank-and-delta-u8-v1" if args.version == "v1"
                        else "cumulative-rank-u16-v1")
            representation = metadata.get("representation", {})
            if representation.get("name") != expected:
                raise ValueError("representation differs from --version")
            if args.version == "v2" and representation.get("scale") != args.scale:
                raise ValueError("scale differs from --scale")
            required = ("rank_u8.npy", "delta_u8.npy") if args.version == "v1" else ("z_u16.npy",)
            if any(name not in archive.namelist() for name in required):
                raise ValueError("missing tensor array")
            if args.verify and not metadata.get("verification", {}).get("passed", False):
                raise ValueError("existing file was not exported with --verify")
    except (OSError, BadZipFile, ValueError, KeyError, AttributeError) as error:
        raise SystemExit(f"Cannot resume {path}: {error}. Use a different output directory or inspect this file.") from error


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

    print(f"Version: {args.version}" + (f"; S={args.scale}" if args.version == "v2" else ""))
    print(f"Found {len(input_files)} IGD files")
    print(f"Output directory: {args.output_dir}")
    start_time = time.monotonic()
    exported = 0
    skipped = 0
    for index, input_path in enumerate(input_files, 1):
        output_path = args.output_dir / f"{input_path.stem}.network.npz"
        if output_path.exists():
            if args.resume:
                validate_existing(output_path, args)
                skipped += 1
                if index % 500 == 0 or index == len(input_files):
                    print(f"Processed {index}/{len(input_files)} files", flush=True)
                continue
            raise SystemExit(
                f"Output already exists: {output_path}. Use --resume to skip existing files."
            )
        command = [
            str(args.extractor_bin),
            str(input_path),
            "-o",
            str(output_path),
        ]
        command.extend(["--network-npz"] if args.version == "v1" else
                       ["--representation", "cumulative-rank", "--scale", str(args.scale)])
        if args.verify:
            command.append("--verify")
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Feature export failed for {input_path.name}:\n{error.stderr}") from error
        exported += 1
        if index % 1000 == 0 or index == len(input_files):
            print(f"Processed {index}/{len(input_files)} files", flush=True)

    elapsed = time.monotonic() - start_time
    print(f"Exported: {exported}; skipped: {skipped}; elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
