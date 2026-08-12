#!/usr/bin/env python3
"""Generate a synthetic stand-in dataset.

Internal dev tool, not a deliverable. The official KLA pairs are not available
yet, so this produces GT + NoisyLR pairs with the structure the real data has
(periodic dies, straight edges, sharp corners, fine lines) using the project's
own degradation engine. It exists so the real CLIs can be exercised end to end;
no metric produced from this data describes the official dataset, and every
figure derived from it is labelled synthetic.

    python scripts/make_fixtures.py --out data/fixtures --count 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kla_restore.degradation import (  # noqa: E402
    DegradationConfig,
    apply_degradations,
    sample_params,
)
from kla_restore.utils import get_logger, save_image_float, setup_logging, write_json  # noqa: E402

LOGGER = get_logger()


def _grid_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Periodic die array: bright pads on a dark field, the dominant wafer motif."""
    img = np.full((size, size), 0.12, dtype=np.float32)
    pitch = int(rng.integers(24, 48))
    pad = max(4, int(pitch * rng.uniform(0.35, 0.6)))
    offset = int(rng.integers(0, pitch))
    for y in range(offset, size, pitch):
        for x in range(offset, size, pitch):
            img[y : y + pad, x : x + pad] = rng.uniform(0.65, 0.95)
    return img


def _line_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Dense line/space grating: the hardest structure for a downsampler to preserve."""
    img = np.full((size, size), 0.15, dtype=np.float32)
    period = int(rng.integers(4, 12))
    width = max(1, period // 2)
    vertical = bool(rng.integers(0, 2))
    for start in range(0, size, period):
        if vertical:
            img[:, start : start + width] = rng.uniform(0.7, 0.98)
        else:
            img[start : start + width, :] = rng.uniform(0.7, 0.98)
    return img


def _trace_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Manhattan routing: axis-aligned traces with right-angle turns."""
    img = np.full((size, size), 0.18, dtype=np.float32)
    for _ in range(int(rng.integers(6, 14))):
        thickness = int(rng.integers(2, 6))
        level = rng.uniform(0.6, 0.95)
        y = int(rng.integers(0, max(1, size - thickness)))
        x = int(rng.integers(0, max(1, size - thickness)))
        length = int(rng.integers(size // 4, size))
        if rng.integers(0, 2):
            img[y : y + thickness, x : min(size, x + length)] = level
            turn = min(size, y + length)
            img[y : turn, x : x + thickness] = level
        else:
            img[y : min(size, y + length), x : x + thickness] = level
            img[y : y + thickness, x : min(size, x + length)] = level
    return img


def _contact_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular contacts/vias, testing curved edges rather than straight ones."""
    img = np.full((size, size), 0.14, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(int(rng.integers(8, 24))):
        cy, cx = rng.uniform(0, size, size=2)
        radius = rng.uniform(3, 12)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        img[mask] = rng.uniform(0.6, 0.97)
    return img


PATTERNS = {
    "grid": _grid_pattern,
    "lines": _line_pattern,
    "traces": _trace_pattern,
    "contacts": _contact_pattern,
}


def make_gt(size: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """One synthetic GT image in the [0, 1] float convention."""
    img = PATTERNS[kind](size, rng)
    # Slow illumination gradient, as real tools exhibit.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    img = img * (0.88 + 0.12 * (yy * rng.uniform(-1, 1) + xx * rng.uniform(-1, 1)))
    return np.clip(img, 0.0, 1.0).astype(np.float32)[:, :, None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GT + NoisyLR fixture pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="data/fixtures", help="output root")
    parser.add_argument("--count", type=int, default=12, help="number of image pairs")
    parser.add_argument("--size", type=int, default=256, help="GT edge length in pixels")
    parser.add_argument("--seed", type=int, default=1234, help="master seed")
    parser.add_argument("--scale", type=int, default=2, help="fixed downsample factor")
    parser.add_argument("--config", default="configs/degradation.yaml", help="degradation config")
    parser.add_argument(
        "--format",
        choices=("png", "npy"),
        default="png",
        help="NoisyLR storage format; npy preserves the unclipped float range, png clips to [0,1]",
    )
    args = parser.parse_args(argv)
    setup_logging(level="INFO")

    out = Path(args.out)
    gt_dir = out / "GT"
    noisy_dir = out / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / args.config
    if config_path.exists():
        from kla_restore.utils import load_yaml

        degradation = DegradationConfig.from_dict(load_yaml(config_path))
    else:
        degradation = DegradationConfig()
    # One fixed scale keeps the fixture set consistent with `--scale` at inference.
    degradation = degradation.with_overrides(scales=(args.scale,), scale_weights=(1.0,))

    rng = np.random.default_rng(args.seed)
    kinds = list(PATTERNS)
    records = []
    total_clipped = 0
    total_pixels = 0
    for index in range(args.count):
        kind = kinds[index % len(kinds)]
        name = f"wafer_{index:03d}_{kind}"
        gt = make_gt(args.size, kind, rng)
        params = sample_params(degradation, seed=args.seed + index)
        noisy = apply_degradations(gt, params)

        # How much signal an 8-bit PNG fixture would destroy, measured not assumed.
        out_of_range = int(np.count_nonzero((noisy < 0.0) | (noisy > 1.0)))
        total_clipped += out_of_range
        total_pixels += int(noisy.size)

        save_image_float(gt_dir / f"{name}_gt.png", gt, bit_depth=8)
        if args.format == "npy":
            np.save(noisy_dir / f"{name}_noisylr.npy", noisy)
        else:
            # NoisyLR legitimately exceeds [0, 1]; an 8-bit PNG cannot hold that, so
            # this fixture is clipped on write. That is a property of the fixture
            # format only. The degradation engine itself never clips.
            save_image_float(noisy_dir / f"{name}_noisylr.png", noisy, bit_depth=8)
        records.append(
            {
                "name": name,
                "pattern": kind,
                "gt_size": [int(gt.shape[0]), int(gt.shape[1])],
                "noisy_size": [int(noisy.shape[0]), int(noisy.shape[1])],
                "order": list(params.order),
                "gaussian_sigma": round(float(params.gaussian_sigma), 6),
                "speckle_sigma": round(float(params.speckle_sigma), 6),
                "scale": int(params.scale),
                "kernel": params.kernel,
                "seed": int(params.seed),
                "noisy_min": round(float(noisy.min()), 6),
                "noisy_max": round(float(noisy.max()), 6),
                "out_of_range_pixels": out_of_range,
            }
        )
        LOGGER.info(
            "%s | %s | order=%s sigma_g=%.4f sigma_s=%.4f x%d %s | range [%.3f, %.3f] oor=%d",
            name,
            f"{gt.shape[0]}x{gt.shape[1]} -> {noisy.shape[0]}x{noisy.shape[1]}",
            "->".join(params.order),
            params.gaussian_sigma,
            params.speckle_sigma,
            params.scale,
            params.kernel,
            float(noisy.min()),
            float(noisy.max()),
            out_of_range,
        )

    clipped_fraction = (total_clipped / total_pixels) if total_pixels else 0.0
    write_json(
        out / "fixtures.json",
        {
            "synthetic": True,
            "warning": "Synthetic stand-in data. Not the official KLA dataset. "
            "No metric computed on it describes official performance.",
            "seed": args.seed,
            "count": args.count,
            "size": args.size,
            "scale": args.scale,
            "noisy_format": args.format,
            "out_of_range_fraction": round(clipped_fraction, 8),
            "degradation": degradation.to_dict(),
            "images": records,
        },
    )
    LOGGER.info(
        "wrote %d pairs to %s (noisy format=%s, %.4f%% of noisy pixels outside [0,1]%s)",
        args.count,
        out.resolve(),
        args.format,
        100.0 * clipped_fraction,
        " and clipped on write" if args.format == "png" else " and preserved",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
