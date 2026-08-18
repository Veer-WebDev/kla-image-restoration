"""Dataset, pairing and splitting.

Key corrections over the starter notebook:

* **Epoch-aware sampling** (audit 3.10). ``samples_per_image`` multiplies the epoch
  length and the per-sample seed includes the epoch, so augmentation is a real
  distribution during training. Validation and test explicitly pin ``epoch=0``, so
  they are frozen and reproducible.
* **Missing official pairs never silently become synthetic** (audit 3.9). Pairing is
  resolved up front into an explicit report; ``on_missing='error'`` is the default
  for evaluation splits.
* **Source-level splitting** with a fixed seed, so no augmented view of a training
  GT can appear in validation or test (audit 1.7, A11).
* **Channel handling is explicit**, never a silent luminance collapse (audit 3.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .degradation import DegradationConfig, degrade, sample_seed
from .extended_degradation import ExtendedDegradationConfig, apply_extended
from .utils import (
    LoadedImage,
    dataloader_generator,
    get_logger,
    image_files,
    load_image_float,
    to_tensor,
    worker_init_fn,
)

LOGGER = get_logger()

#: Suffixes stripped from a filename stem when matching a NoisyLR file to its GT.
NOISY_TOKENS: tuple[str, ...] = (
    "_noisylr",
    "_noisy_lr",
    "_noisy",
    "_lr",
    "_degraded",
    "_input",
    "-noisylr",
    "-noisy",
    "-lr",
)

#: Suffixes stripped from a GT filename stem.
GT_TOKENS: tuple[str, ...] = ("_gt", "_groundtruth", "_ground_truth", "_hr", "_clean", "-gt", "-hr")

_TRAILING_INDEX = re.compile(r"[_-]?(x\d+|scale\d+)$", re.IGNORECASE)


def canonical_stem(path: Path, tokens: Sequence[str]) -> str:
    """Reduce a filename to a pairing key by stripping known role suffixes."""
    stem = path.stem
    lowered = stem.lower()
    for token in tokens:
        if lowered.endswith(token):
            stem = stem[: len(stem) - len(token)]
            lowered = stem.lower()
            break
    stem = _TRAILING_INDEX.sub("", stem)
    return stem.lower()


@dataclass
class PairReport:
    """Outcome of pairing GT files with NoisyLR files."""

    paired: dict[str, tuple[Path, Path]] = field(default_factory=dict)
    gt_only: list[str] = field(default_factory=list)
    noisy_only: list[str] = field(default_factory=list)
    duplicate_gt: list[str] = field(default_factory=list)
    duplicate_noisy: list[str] = field(default_factory=list)

    @property
    def n_paired(self) -> int:
        return len(self.paired)

    def summary(self) -> dict[str, Any]:
        return {
            "paired": self.n_paired,
            "gt_only": len(self.gt_only),
            "noisy_only": len(self.noisy_only),
            "duplicate_gt": len(self.duplicate_gt),
            "duplicate_noisy": len(self.duplicate_noisy),
        }

    def log(self) -> None:
        LOGGER.info(
            "pairing | paired=%d gt_only=%d noisy_only=%d dup_gt=%d dup_noisy=%d",
            self.n_paired,
            len(self.gt_only),
            len(self.noisy_only),
            len(self.duplicate_gt),
            len(self.duplicate_noisy),
        )
        for key in self.noisy_only[:10]:
            LOGGER.warning("pairing | NoisyLR without GT: %s", key)
        for key in self.duplicate_gt[:10]:
            LOGGER.warning("pairing | ambiguous GT key: %s", key)


def discover_pairs(
    gt_dir: str | Path,
    noisy_dir: str | Path | None = None,
) -> tuple[dict[str, Path], dict[str, Path], PairReport]:
    """Discover GT files and optional NoisyLR partners.

    Returns
    -------
    (gt_map, noisy_map, report)
        ``gt_map`` maps a canonical stem to the GT path, ``noisy_map`` likewise for
        NoisyLR. ``report`` records every unpaired or ambiguous file.
    """
    report = PairReport()
    gt_map: dict[str, Path] = {}
    for path in image_files(gt_dir):
        key = canonical_stem(path, GT_TOKENS)
        if key in gt_map:
            report.duplicate_gt.append(key)
            continue
        gt_map[key] = path

    noisy_map: dict[str, Path] = {}
    if noisy_dir is not None:
        for path in image_files(noisy_dir):
            key = canonical_stem(path, NOISY_TOKENS)
            if key in noisy_map:
                report.duplicate_noisy.append(key)
                continue
            noisy_map[key] = path

    for key, gt_path in gt_map.items():
        if key in noisy_map:
            report.paired[key] = (gt_path, noisy_map[key])
        else:
            report.gt_only.append(key)
    report.noisy_only = sorted(set(noisy_map) - set(gt_map))
    return gt_map, noisy_map, report


def split_keys(
    keys: Iterable[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split source keys into train/val/test at **source level**.

    Deterministic for a given ``(keys, ratios, seed)``: keys are sorted before
    shuffling so filesystem ordering cannot influence the split.
    """
    keys = sorted(set(keys))
    if not keys:
        return {"train": [], "val": [], "test": []}
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError(f"split ratios must sum to a positive value, got {ratios}")
    ratios = tuple(r / total for r in ratios)  # type: ignore[assignment]

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(keys))
    shuffled = [keys[i] for i in order]

    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Guarantee non-empty val/test whenever the data allows it.
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _random_crop_box(height: int, width: int, size: int, rng: np.random.Generator) -> tuple[int, int]:
    top = int(rng.integers(0, max(1, height - size + 1)))
    left = int(rng.integers(0, max(1, width - size + 1)))
    return top, left


def _apply_flips_rot(arr: np.ndarray, code: int) -> np.ndarray:
    """Apply one of eight dihedral transforms (identity included)."""
    out = arr
    if code & 1:
        out = out[:, ::-1, :]
    if code & 2:
        out = out[::-1, :, :]
    if code & 4:
        out = out.transpose(1, 0, 2)
    return np.ascontiguousarray(out)


@dataclass
class DatasetConfig:
    """Dataset behaviour, fully serializable into the run snapshot."""

    patch_size: int = 256
    samples_per_image: int = 8
    synthetic_prob: float = 0.6
    augment: bool = True
    augment_flips: bool = True
    augment_rot90: bool = True
    grayscale: bool | None = None
    channels: int = 1
    training: bool = True
    max_eval_size: int | None = None
    cache_images: bool = True

    def __post_init__(self) -> None:
        if self.patch_size < 16:
            raise ValueError(f"patch_size must be >= 16, got {self.patch_size}")
        if self.samples_per_image < 1:
            raise ValueError(f"samples_per_image must be >= 1, got {self.samples_per_image}")
        if not 0.0 <= self.synthetic_prob <= 1.0:
            raise ValueError(f"synthetic_prob must lie in [0, 1], got {self.synthetic_prob}")
        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "samples_per_image": self.samples_per_image,
            "synthetic_prob": self.synthetic_prob,
            "augment": self.augment,
            "augment_flips": self.augment_flips,
            "augment_rot90": self.augment_rot90,
            "grayscale": self.grayscale,
            "channels": self.channels,
            "training": self.training,
            "max_eval_size": self.max_eval_size,
            "cache_images": self.cache_images,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)  # noqa: SLF001
        if unknown:
            raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
        return cls(**data)


class RestorationDataset(Dataset):
    """Paired (NoisyLR, GT) samples, either official or synthetically degraded.

    Modes
    -----
    ``synthetic``
        Degrade the GT on the fly with :mod:`kla_restore.degradation`.
    ``official``
        Use the provided NoisyLR file. Requires a paired file for every key.
    ``mixed``
        Per-sample coin flip with probability ``synthetic_prob`` of going synthetic;
        the flip is seeded, and the realized counts are reported by
        :meth:`consumption_stats`.
    """

    def __init__(
        self,
        keys: Sequence[str],
        gt_map: dict[str, Path],
        noisy_map: dict[str, Path] | None = None,
        *,
        mode: Literal["synthetic", "official", "mixed"] = "synthetic",
        degradation: DegradationConfig | None = None,
        extended: ExtendedDegradationConfig | None = None,
        config: DatasetConfig | None = None,
        seed: int = 42,
        on_missing: Literal["error", "synthetic", "skip"] = "error",
    ) -> None:
        self.config = config or DatasetConfig()
        self.degradation = degradation or DegradationConfig()
        self.extended = extended or ExtendedDegradationConfig()
        self.gt_map = dict(gt_map)
        self.noisy_map = dict(noisy_map or {})
        self.mode = mode
        self.seed = int(seed)
        self.on_missing = on_missing
        self._epoch = 0
        self._cache: dict[str, LoadedImage] = {}
        self._counts = {"synthetic": 0, "official": 0}

        resolved: list[str] = []
        missing: list[str] = []
        for key in keys:
            if key not in self.gt_map:
                raise KeyError(f"key {key!r} has no GT file")
            if mode in {"official", "mixed"} and key not in self.noisy_map:
                missing.append(key)
                if mode == "official":
                    if on_missing == "error":
                        continue
                    if on_missing == "skip":
                        continue
            resolved.append(key)

        if missing:
            if mode == "official" and on_missing == "error":
                raise FileNotFoundError(
                    f"{len(missing)} key(s) lack a NoisyLR partner in official mode: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                    "Pass on_missing='synthetic' to substitute synthetic samples explicitly, "
                    "or on_missing='skip' to drop them."
                )
            LOGGER.warning(
                "dataset | %d/%d keys lack a NoisyLR partner; on_missing=%s",
                len(missing),
                len(keys),
                on_missing,
            )
        if not resolved:
            raise ValueError("dataset is empty after pairing resolution")

        self.keys: list[str] = sorted(resolved)
        self.missing_official: list[str] = sorted(missing)
        LOGGER.info(
            "dataset | mode=%s keys=%d samples_per_image=%d length=%d",
            mode,
            len(self.keys),
            self.config.samples_per_image if self.config.training else 1,
            len(self),
        )

    # ------------------------------------------------------------------ plumbing
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used in per-sample seeds (training only)."""
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    def consumption_stats(self) -> dict[str, int]:
        """Realized synthetic/official sample counts since construction."""
        return dict(self._counts)

    def reset_stats(self) -> None:
        self._counts = {"synthetic": 0, "official": 0}

    def __len__(self) -> int:
        per_image = self.config.samples_per_image if self.config.training else 1
        return len(self.keys) * per_image

    def _load(self, path: Path, *, clip: bool) -> LoadedImage:
        cache_key = f"{path}|{int(clip)}"
        if self.config.cache_images and cache_key in self._cache:
            return self._cache[cache_key]
        loaded = load_image_float(
            path,
            clip=clip,
            grayscale=self.config.grayscale if self.config.grayscale is not None else None,
        )
        if self.config.channels == 1 and loaded.channels == 3:
            loaded = load_image_float(path, clip=clip, grayscale=True)
        elif self.config.channels == 3 and loaded.channels == 1:
            loaded = LoadedImage(
                array=np.repeat(loaded.array, 3, axis=2),
                source_dtype=loaded.source_dtype,
                scale_divisor=loaded.scale_divisor,
                clipped=loaded.clipped,
                path=loaded.path,
            )
        if self.config.cache_images:
            self._cache[cache_key] = loaded
        return loaded

    # ------------------------------------------------------------------- getitem
    def __getitem__(self, index: int) -> dict[str, Any]:
        per_image = self.config.samples_per_image if self.config.training else 1
        key = self.keys[index // per_image]
        sample_index = index % per_image
        epoch = self._epoch if self.config.training else 0
        seed = sample_seed(self.seed, key, sample_index, epoch)
        rng = np.random.default_rng(seed % (2**63))

        gt = self._load(self.gt_map[key], clip=True).array

        use_official = False
        if self.mode == "official" and key in self.noisy_map:
            use_official = True
        elif self.mode == "mixed" and key in self.noisy_map:
            use_official = rng.random() >= self.config.synthetic_prob

        if use_official:
            noisy = self._load(self.noisy_map[key], clip=False).array
            gt, noisy = self._crop_official(gt, noisy, rng)
            params_dict: dict[str, Any] = {"order": "official", "scale": 0, "kernel": "official"}
            self._counts["official"] += 1
        else:
            gt = self._crop_gt(gt, rng)
            noisy, params = degrade(gt, self.degradation, int(seed))
            params_dict = params.to_dict()
            self._counts["synthetic"] += 1

        # Opt-in extended SEM artifacts (robustness augmentation). Applied only in
        # training mode, on the NoisyLR, and only when the config enables it. This
        # never touches the KLA-faithful evaluation path.
        if self.config.training and self.extended.any_active:
            noisy, extended_params = apply_extended(noisy, self.extended, int(seed))
            if extended_params:
                params_dict = {**params_dict, "extended": extended_params}

        if self.config.training and self.config.augment:
            code = int(rng.integers(0, 8))
            if not self.config.augment_flips:
                code &= 4
            if not self.config.augment_rot90:
                code &= 3
            if code:
                gt = _apply_flips_rot(gt, code)
                noisy = _apply_flips_rot(noisy, code)

        gt_t = to_tensor(gt)
        noisy_t = to_tensor(noisy)
        scale_h = gt_t.shape[-2] / max(1, noisy_t.shape[-2])
        return {
            "noisy": noisy_t,
            "gt": gt_t,
            "key": key,
            "sample_index": int(sample_index),
            "seed": int(seed),
            "scale": float(scale_h),
            "source": "official" if use_official else "synthetic",
            "params": params_dict,
        }

    # -------------------------------------------------------------------- crops
    def _crop_gt(self, gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Crop the GT for synthetic degradation.

        The crop size is forced to a multiple of the largest configured scale so
        integer downsampling is exact and no fractional alignment is introduced.
        """
        if not self.config.training:
            return self._limit_eval_size(gt)
        multiple = max(self.degradation.scales)
        size = self.config.patch_size - (self.config.patch_size % multiple)
        size = max(multiple, min(size, gt.shape[0], gt.shape[1]))
        size -= size % multiple
        size = max(multiple, size)
        top, left = _random_crop_box(gt.shape[0], gt.shape[1], size, rng)
        return np.ascontiguousarray(gt[top : top + size, left : left + size])

    def _crop_official(
        self, gt: np.ndarray, noisy: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop an official pair consistently in both resolutions."""
        ratio_h = gt.shape[0] / max(1, noisy.shape[0])
        ratio_w = gt.shape[1] / max(1, noisy.shape[1])
        ratio = int(round(ratio_h))
        if ratio < 1 or abs(ratio_h - ratio) > 0.05 or abs(ratio_w - ratio) > 0.05:
            LOGGER.debug(
                "official pair with non-integer ratio (%.3f, %.3f); using full images",
                ratio_h,
                ratio_w,
            )
            return self._limit_eval_size(gt), noisy
        if not self.config.training:
            return self._limit_eval_size(gt), self._limit_eval_size(noisy, ratio)

        lr_size = max(8, self.config.patch_size // ratio)
        lr_size = min(lr_size, noisy.shape[0], noisy.shape[1])
        top, left = _random_crop_box(noisy.shape[0], noisy.shape[1], lr_size, rng)
        noisy_crop = noisy[top : top + lr_size, left : left + lr_size]
        gt_crop = gt[
            top * ratio : (top + lr_size) * ratio,
            left * ratio : (left + lr_size) * ratio,
        ]
        return np.ascontiguousarray(gt_crop), np.ascontiguousarray(noisy_crop)

    def _limit_eval_size(self, arr: np.ndarray, divisor: int = 1) -> np.ndarray:
        """Optionally centre-crop huge evaluation images to bound memory."""
        limit = self.config.max_eval_size
        if not limit:
            return arr
        limit = max(1, limit // max(1, divisor))
        h, w = arr.shape[:2]
        if h <= limit and w <= limit:
            return arr
        top = max(0, (h - limit) // 2)
        left = max(0, (w - limit) // 2)
        return np.ascontiguousarray(arr[top : top + limit, left : left + limit])


def collate_variable(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch, stacking tensors only when all shapes agree.

    Evaluation batches full images of differing sizes; those are returned as lists
    so the caller can loop without a crash.
    """
    noisy = [item["noisy"] for item in batch]
    gt = [item["gt"] for item in batch]
    same = len({tuple(t.shape) for t in noisy}) == 1 and len({tuple(t.shape) for t in gt}) == 1
    out: dict[str, Any] = {
        "keys": [item["key"] for item in batch],
        "seeds": [item["seed"] for item in batch],
        "sources": [item["source"] for item in batch],
        "params": [item["params"] for item in batch],
        "scales": [item["scale"] for item in batch],
        "stacked": same,
    }
    if same:
        out["noisy"] = torch.stack(noisy)
        out["gt"] = torch.stack(gt)
    else:
        out["noisy"] = noisy
        out["gt"] = gt
    return out


def build_dataloader(
    dataset: RestorationDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with deterministic worker seeding (audit 3.8)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_variable,
        generator=dataloader_generator(seed),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def detect_channels(gt_map: dict[str, Path], limit: int = 8) -> int:
    """Inspect a few GT files and report the dominant channel count (audit 3.1)."""
    counts: dict[int, int] = {}
    for path in list(gt_map.values())[:limit]:
        try:
            channels = load_image_float(path, clip=True).channels
        except Exception as exc:  # pragma: no cover - corrupt file
            LOGGER.warning("channel detection failed for %s: %s", path, exc)
            continue
        counts[channels] = counts.get(channels, 0) + 1
    if not counts:
        return 1
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    if len(counts) > 1:
        LOGGER.warning("mixed channel counts across GT files: %s; using %d", counts, dominant)
    return 3 if dominant >= 3 else 1


def manifest(dataset: RestorationDataset) -> dict[str, Any]:
    """Serializable description of a dataset split, written next to checkpoints."""
    return {
        "mode": dataset.mode,
        "seed": dataset.seed,
        "n_keys": len(dataset.keys),
        "length": len(dataset),
        "keys": list(dataset.keys),
        "missing_official": list(dataset.missing_official),
        "dataset_config": dataset.config.to_dict(),
        "degradation_config": dataset.degradation.to_dict(),
    }
