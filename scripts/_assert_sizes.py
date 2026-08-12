#!/usr/bin/env python3
"""Assert restored outputs match the scale contract. Internal dev tool.

Checks that every NoisyLR input produced exactly one output at scale x2 of the
input size, and that the output is not a trivial constant image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCALE = 2


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _assert_sizes.py OUTPUT_DIR [INPUT_DIR]")
        return 2
    out = Path(argv[1])
    src_dir = Path(argv[2]) if len(argv) > 2 else Path("data/fixtures/NoisyLR")

    inputs = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".npy"})
    if not inputs:
        print(f"no inputs found in {src_dir}")
        return 2

    bad = 0
    for src in inputs:
        dst = out / f"{src.stem}.png"
        if not dst.exists():
            print(f"MISSING {dst}")
            bad += 1
            continue
        if src.suffix.lower() == ".npy":
            arr = np.load(src)
            h0, w0 = arr.shape[:2]
        else:
            w0, h0 = Image.open(src).size
        with Image.open(dst) as img:
            w1, h1 = img.size
            data = np.asarray(img, dtype=np.float32)
        size_ok = (w1, h1) == (w0 * SCALE, h0 * SCALE)
        spread = float(data.max() - data.min())
        varied_ok = spread > 1.0  # 8-bit levels; a constant image would fail
        if not (size_ok and varied_ok):
            bad += 1
        print(
            f"{'ok  ' if size_ok and varied_ok else 'BAD '} {src.name}: "
            f"{w0}x{h0} -> {w1}x{h1} (expected {w0 * SCALE}x{h0 * SCALE}), spread={spread:.1f}"
        )

    produced = len(list(out.glob("*.png")))
    print(f"count in={len(inputs)} out={produced} bad={bad}")
    if produced != len(inputs):
        print(f"FAIL output count {produced} != input count {len(inputs)}")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
