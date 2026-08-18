#!/usr/bin/env python3
"""Materialize a source-disjoint synthetic restoration corpus.

The KLA restoration brief permits only additive Gaussian noise, multiplicative
speckle noise and downsampling.  This script creates auditable GT/NoisyLR pairs
from a directory of clean, disclosed source images.  It never uses an image
from one source hash in more than one split.

NoisyLR arrays are saved as ``.npy`` so values outside ``[0, 1]`` survive the
on-disk boundary.  The repository loader and evaluator deliberately support
that format.  GT images are clipped and saved as 8-bit PNGs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kla_restore.dataset import split_keys  # noqa: E402
from kla_restore.degradation import (  # noqa: E402
    ORDER_PERMUTATIONS,
    DegradationConfig,
    apply_degradations,
    sample_params,
    sample_seed,
)
from kla_restore.utils import image_files, load_image_float, save_image_float  # noqa: E402

SPLITS = ("train", "val", "test")
FIELDNAMES = [
    "sample_id",
    "source_file",
    "source_sha256",
    "split",
    "view_index",
    "seed",
    "crop_top",
    "crop_left",
    "gaussian_sigma",
    "speckle_sigma",
    "scale",
    "kernel",
    "order",
    "gt_path",
    "noisylr_path",
    "noisylr_encoding",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_float_gray(array: np.ndarray, size: int) -> np.ndarray:
    """Resize a float grayscale image without changing its numeric convention."""
    plane = np.asarray(array, dtype=np.float32)
    if plane.ndim == 3:
        plane = plane[:, :, 0]
    image = Image.fromarray(plane, mode="F")
    resized = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)[:, :, None]


def _crop_or_resize(array: np.ndarray, crop_size: int, seed: int) -> tuple[np.ndarray, int, int]:
    """Return a deterministic square crop, resizing only when the source is too small."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    height, width = arr.shape[:2]
    if height < crop_size or width < crop_size:
        return _resize_float_gray(arr, crop_size), 0, 0
    rng = np.random.default_rng(seed % (2**63))
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    return np.ascontiguousarray(arr[top : top + crop_size, left : left + crop_size]), top, left


def _order_code(order: Iterable[str]) -> str:
    return "".join(item[0].upper() for item in order)


def _materializer_config(*, scale: int, order: tuple[str, str, str]) -> DegradationConfig:
    return DegradationConfig(
        gaussian_sigma=(0.005, 0.08),
        speckle_sigma=(0.01, 0.15),
        scales=(int(scale),),
        scale_weights=(1.0,),
        kernels=("area", "bicubic", "bilinear"),
        fixed_order=order,
        clip_noisy=False,
    )


def materialize(
    *,
    source_dir: Path,
    out_dir: Path,
    seed: int,
    views_per_source: int,
    crop_size: int,
    scale: int,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, object]:
    """Create source-disjoint GT/NoisyLR directories and auditable manifests.

    Parameters are deliberately small and explicit so the exact corpus is
    reproducible from the source files and a manifest.  A source is identified
    by SHA-256 of its original file bytes before any crop, resize or degradation.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if views_per_source < 1:
        raise ValueError("views_per_source must be >= 1")
    if crop_size < 16 or crop_size % int(scale) != 0:
        raise ValueError("crop_size must be >= 16 and divisible by scale")
    if int(scale) < 1:
        raise ValueError("scale must be >= 1")

    source_paths = image_files(source_dir)
    if len(source_paths) < 3:
        raise ValueError("at least three source images are required for train/val/test splits")

    source_records = [(path, _sha256(path)) for path in source_paths]
    hashes = [digest for _, digest in source_records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate source image bytes detected; remove duplicate source files")
    source_splits = split_keys(hashes, ratios=split_ratios, seed=int(seed))
    split_for_hash = {digest: split for split, values in source_splits.items() for digest in values}

    writers: dict[str, tuple[object, csv.DictWriter]] = {}
    for split in SPLITS:
        (out_dir / split / "GT").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "NoisyLR").mkdir(parents=True, exist_ok=True)
        handle = (out_dir / f"{split}_manifest.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writers[split] = (handle, writer)

    records_by_split = {split: 0 for split in SPLITS}
    observed_orders: set[str] = set()
    try:
        for source_path, source_hash in sorted(source_records, key=lambda item: item[1]):
            split = split_for_hash[source_hash]
            source = load_image_float(source_path, clip=True, grayscale=True).array
            handle, writer = writers[split]
            del handle
            for view_index in range(int(views_per_source)):
                view_seed = sample_seed(int(seed), source_hash, view_index)
                gt, crop_top, crop_left = _crop_or_resize(source, int(crop_size), view_seed)
                order = ORDER_PERMUTATIONS[view_index % len(ORDER_PERMUTATIONS)]
                config = _materializer_config(scale=int(scale), order=order)
                params = sample_params(config, view_seed)
                noisy = apply_degradations(gt, params, clip=False)
                order_code = _order_code(params.order)
                sample_id = f"{source_hash[:12]}_v{view_index:02d}"
                gt_name = f"{sample_id}_gt.png"
                noisy_name = f"{sample_id}_noisylr.npy"
                gt_path = out_dir / split / "GT" / gt_name
                noisy_path = out_dir / split / "NoisyLR" / noisy_name
                save_image_float(gt_path, np.clip(gt, 0.0, 1.0), bit_depth=8)
                np.save(noisy_path, noisy.astype(np.float32))
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "source_file": source_path.name,
                        "source_sha256": source_hash,
                        "split": split,
                        "view_index": view_index,
                        "seed": view_seed,
                        "crop_top": crop_top,
                        "crop_left": crop_left,
                        "gaussian_sigma": f"{params.gaussian_sigma:.8f}",
                        "speckle_sigma": f"{params.speckle_sigma:.8f}",
                        "scale": params.scale,
                        "kernel": params.kernel,
                        "order": order_code,
                        "gt_path": gt_path.relative_to(out_dir).as_posix(),
                        "noisylr_path": noisy_path.relative_to(out_dir).as_posix(),
                        "noisylr_encoding": "float32_npy_unclipped",
                    }
                )
                records_by_split[split] += 1
                observed_orders.add(order_code)
    finally:
        for handle, _ in writers.values():
            handle.close()  # type: ignore[union-attr]

    source_sets = {split: set(values) for split, values in source_splits.items()}
    source_sets_disjoint = all(
        source_sets[left].isdisjoint(source_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    )
    summary: dict[str, object] = {
        "synthetic": True,
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "n_sources": len(source_records),
        "source_sha256": {path.name: digest for path, digest in source_records},
        "split_source_counts": {split: len(source_splits[split]) for split in SPLITS},
        "split_sample_counts": records_by_split,
        "source_sets_disjoint": source_sets_disjoint,
        "views_per_source": int(views_per_source),
        "crop_size": int(crop_size),
        "scale": int(scale),
        "orders": sorted(observed_orders),
        "noisylr_encoding": "float32_npy_unclipped",
        "warning": "Synthetic restoration corpus. It is not official KLA data and does not establish hidden-test performance.",
    }
    with (out_dir / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _parse_ratios(raw: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in raw.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("split ratios must be three comma-separated values")
    if sum(values) <= 0 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("split ratios must be non-negative with positive total")
    return values  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--views-per-source", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--split-ratios", type=_parse_ratios, default=(0.8, 0.1, 0.1))
    args = parser.parse_args(argv)
    summary = materialize(
        source_dir=args.source_dir,
        out_dir=args.out,
        seed=args.seed,
        views_per_source=args.views_per_source,
        crop_size=args.crop_size,
        scale=args.scale,
        split_ratios=args.split_ratios,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
