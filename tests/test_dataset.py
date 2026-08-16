from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from kla_restore.dataset import discover_pairs, split_keys
from kla_restore.utils import load_image_float


def _png(path: Path, value: int, size: tuple[int, int] = (9, 11)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size, value, dtype=np.uint8)).save(path)


def test_discover_pairs_recursively_and_strips_official_role_suffixes(tmp_path: Path) -> None:
    gt = tmp_path / "GT"
    noisy = tmp_path / "NoisyLR"
    _png(gt / "batch_a" / "wafer_001_gt.png", 255)
    _png(noisy / "batch_b" / "wafer_001_noisylr.png", 100)

    gt_map, noisy_map, report = discover_pairs(gt, noisy)

    assert set(gt_map) == {"wafer_001"}
    assert set(noisy_map) == {"wafer_001"}
    assert report.n_paired == 1
    assert report.gt_only == []
    assert report.noisy_only == []


def test_discover_pairs_reports_missing_and_duplicate_stems(tmp_path: Path) -> None:
    gt = tmp_path / "GT"
    noisy = tmp_path / "NoisyLR"
    _png(gt / "wafer_001_gt.png", 1)
    _png(gt / "copy" / "wafer_001_clean.png", 2)
    _png(gt / "wafer_002_gt.png", 3)
    _png(noisy / "wafer_003_noisylr.png", 4)

    _, _, report = discover_pairs(gt, noisy)

    assert report.duplicate_gt == ["wafer_001"]
    assert report.gt_only == ["wafer_001", "wafer_002"]
    assert report.noisy_only == ["wafer_003"]
    assert report.n_paired == 0


def test_source_split_is_deterministic_and_has_no_leakage() -> None:
    keys = [f"source_{i:03d}" for i in range(21)]
    first = split_keys(keys, seed=42)
    second = split_keys(reversed(keys), seed=42)

    assert first == second
    train, val, test = (set(first[name]) for name in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(keys)


def test_image_normalization_preserves_noisylr_out_of_range_when_unclipped(tmp_path: Path) -> None:
    raw = np.array([[0.25, 1.15]], dtype=np.float32)
    path = tmp_path / "noisy.npy"
    np.save(path, raw)

    noisy = load_image_float(path, clip=False)
    gt = load_image_float(path, clip=True)

    assert noisy.array.shape == (1, 2, 1)
    assert noisy.array.dtype == np.float32
    assert float(noisy.array.max()) == np.float32(1.15)
    assert float(gt.array.max()) == 1.0
