"""Convert RGB PNGs to the exact 8-bit YCoCg-R representation of gdcompress.

PNG's R/G/B slots contain Y/biased Co/biased Cg, not display RGB colors.
All lifting steps wrap modulo 256, matching image_processor.rs. This differs
from conventional YCoCg-R with signed 9-bit chroma; do not clip the chroma.
"""

import argparse
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from PIL import Image


DATASET_PATH = "data/cifar-10_resized/train"
OUTPUT_DIR = "data/cifar-10_resized_YCoCgR/train"


def _signed_half(value):
    """Interpret a wrapped byte as i8, then arithmetic-shift right by one."""
    return np.where(value < 128, value, value - 256) >> 1


def _channels(image):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError("Expected a uint8 array with shape (..., 3)")
    return np.moveaxis(array.astype(np.int16), -1, 0)


def rgb_to_ycocgr(rgb):
    """Return uint8 channels [Y, (Co + 128) % 256, (Cg + 128) % 256]."""
    r, g, b = _channels(rgb)
    co = (r - b) % 256
    t = (b + _signed_half(co)) % 256
    cg = (g - t) % 256
    y = (t + _signed_half(cg)) % 256
    return np.stack((y, (co + 128) % 256, (cg + 128) % 256), axis=-1).astype(np.uint8)


def ycocgr_to_rgb(ycocgr):
    """Invert the wrapped representation exactly, without quantization."""
    y, biased_co, biased_cg = _channels(ycocgr)
    co = (biased_co - 128) % 256
    cg = (biased_cg - 128) % 256
    t = (y - _signed_half(cg)) % 256
    g = (cg + t) % 256
    b = (t - _signed_half(co)) % 256
    r = (b + co) % 256
    return np.stack((r, g, b), axis=-1).astype(np.uint8)


def convert_dataset(input_dir, output_dir):
    """Convert PNGs, retaining names and any class subdirectories."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the input directory")

    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
    start_time = timer()
    for count, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"))
        converted = rgb_to_ycocgr(rgb)
        output_path = output_dir / path.relative_to(input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(converted).save(output_path, format="PNG")
        if count % 1000 == 0:
            print(f"  {count} images processed")

    print(f"Processing completed! Total images: {len(paths)}")
    print(f"Processing time: {timer() - start_time:.1f}s")
    print(f"Images saved to: {output_dir}")
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DATASET_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
