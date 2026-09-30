"""Build the first compressed-domain dataset: PNG -> IGD 1x1 -> uint8 NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 1x1 IGD files and their rank/delta uint8 NPZ representations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_resized",
        help="Directory containing PNG source images.",
    )
    parser.add_argument(
        "--igd-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10_compressed_pg_1x1",
        help="Directory that retains the compressed IGD files.",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "network_features_1x1_v1",
        help="Directory for final rank_u8/delta_u8 NPZ files.",
    )
    parser.add_argument(
        "--gdcompress-bin",
        type=Path,
        default=PROJECT_ROOT / "src" / "gdcompress" / "target" / "release" / "gdcompress.exe",
    )
    parser.add_argument(
        "--extractor-bin",
        type=Path,
        default=PROJECT_ROOT / "src" / "gdcompress" / "target" / "release" / "igd_extract.exe",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify every image after feature extraction; use first on a small sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing IGD/NPZ files and resume unfinished work.",
    )
    return parser.parse_args()


def run(command: list[str], description: str) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{description}: {error}") from error


def main() -> None:
    args = parse_args()
    for required, description in [
        (args.input_dir, "Input directory"),
        (args.gdcompress_bin, "gdcompress executable"),
        (args.extractor_bin, "igd_extract executable"),
    ]:
        if not required.exists():
            raise SystemExit(f"{description} not found: {required}")
    input_files = sorted(args.input_dir.glob("*.png"))
    if not input_files:
        raise SystemExit(f"No PNG files found in: {args.input_dir}")
    args.igd_dir.mkdir(parents=True, exist_ok=True)
    args.npz_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    completed = 0
    reused_igd = 0
    skipped_npz = 0
    for index, image_path in enumerate(input_files, 1):
        igd_path = args.igd_dir / f"{image_path.stem}.igd"
        npz_path = args.npz_dir / f"{image_path.stem}.network.npz"
        if npz_path.exists():
            if args.resume:
                skipped_npz += 1
                continue
            raise SystemExit(f"NPZ already exists: {npz_path}. Use --resume to skip it.")
        if igd_path.exists():
            if not args.resume:
                raise SystemExit(f"IGD already exists: {igd_path}. Use --resume to reuse it.")
            reused_igd += 1
        else:
            run(
                [
                    str(args.gdcompress_bin),
                    str(image_path),
                    "--pixel-grouping",
                    "1x1",
                    "-o",
                    str(igd_path),
                ],
                f"Compression failed for {image_path.name}",
            )
        command = [
            str(args.extractor_bin),
            str(igd_path),
            "--network-npz",
            "-o",
            str(npz_path),
        ]
        if args.verify:
            command.append("--verify")
        run(command, f"NPZ export failed for {image_path.name}")
        completed += 1
        if index % 500 == 0 or index == len(input_files):
            print(f"Processed {index}/{len(input_files)} images")

    elapsed = time.monotonic() - start
    print(
        f"Created NPZ: {completed}; reused IGD: {reused_igd}; "
        f"skipped NPZ: {skipped_npz}; elapsed: {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
