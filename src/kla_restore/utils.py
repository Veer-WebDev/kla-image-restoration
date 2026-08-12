"""Shared utilities: seeding, determinism, logging, image IO, config handling.

Design rules enforced here:

* Normalization is **dtype driven** and decided *before* any channel arithmetic
  (audit finding 3.2). The scale factor used for every image is recoverable.
* GT is clipped to [0, 1]; NoisyLR is never clipped (audit findings 1.5, 1.6).
* Nothing in this module imports torch-optional evaluation-only packages
  (lpips, skimage, matplotlib, pandas). ``inference.py`` depends on this module,
  so it must stay lean: torch + numpy + Pillow only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

LOGGER_NAME = "kla_restore"

#: Image suffixes considered when discovering dataset files.
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm", ".npy"}
)

#: Readable alias used by the inference entry point.
IMAGE_EXTENSIONS = IMAGE_EXTS

#: Luminance weights (ITU-R BT.601) used only when an explicit grayscale
#: conversion is requested by configuration -- never silently (audit finding 3.1).
_BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------
def setup_logging(
    log_file: str | os.PathLike[str] | None = None,
    *,
    level: int | str = logging.INFO,
    verbosity: int | None = None,
) -> logging.Logger:
    """Configure and return the package logger.

    Calling this again adds a file handler without discarding an already
    configured stream handler, so ``setup_logging()`` at CLI start followed by
    ``setup_logging(run_dir / 'train.log')`` once the run directory is known both
    take effect.

    Parameters
    ----------
    log_file:
        Optional path; the directory is created if needed. Logs are appended.
    level:
        Logging level as an int or a name such as ``"DEBUG"``.
    verbosity:
        Legacy integer form. ``0`` -> WARNING, ``1`` -> INFO, ``>=2`` -> DEBUG.
        Takes precedence over ``level`` when given.
    """
    if verbosity is not None:
        resolved = {0: logging.WARNING, 1: logging.INFO}.get(int(verbosity), logging.DEBUG)
    elif isinstance(level, str):
        resolved = getattr(logging, level.strip().upper(), logging.INFO)
    else:
        resolved = int(level)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(fmt)
        stream.setLevel(resolved)
        logger.addHandler(stream)
    else:
        for handler in logger.handlers:
            handler.setLevel(resolved)

    if log_file is not None:
        path = Path(log_file).resolve()
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
            for h in logger.handlers
        )
        if not already:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            file_handler.setFormatter(fmt)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the package logger, adding a null handler if unconfigured."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
def seed_everything(seed: int = 42, strict: bool = False) -> int:
    """Seed python, numpy and torch.

    Parameters
    ----------
    seed:
        Master seed.
    strict:
        When ``True`` also request deterministic CUDA kernels via
        ``torch.use_deterministic_algorithms`` and set ``CUBLAS_WORKSPACE_CONFIG``.
        Strict mode is slower and can raise for ops without deterministic
        implementations, which is why it is opt-in (audit finding 3.7).

    Returns
    -------
    int
        The seed that was applied, for logging convenience.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - torch version dependent
            get_logger().warning("strict determinism unavailable: %s", exc)
    return seed


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for ``DataLoader(generator=...)``."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically (audit finding 3.8)."""
    base = torch.initial_seed() % (2**32 - 1)
    seed = (base + worker_id) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(*parts: Any) -> int:
    """Derive a stable 63-bit seed from arbitrary hashable parts.

    Uses BLAKE2b over the repr of the parts so results are stable across
    processes and platforms, unlike Python's salted ``hash``.
    """
    payload = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def select_device(requested: str = "auto") -> torch.device:
    """Resolve a device string, falling back to CPU when CUDA is unavailable."""
    requested = (requested or "auto").lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        get_logger().warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------------------
# image IO
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedImage:
    """An image loaded as float32 together with the provenance of its scaling."""

    array: np.ndarray  # (H, W, C) float32
    source_dtype: str
    scale_divisor: float
    clipped: bool
    path: str

    @property
    def channels(self) -> int:
        return int(self.array.shape[2])

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.array.shape[0]), int(self.array.shape[1])


def _dtype_divisor(dtype: np.dtype, sample_max: float, assume_8bit_floats: bool) -> float:
    """Return the divisor mapping raw values onto the [0, 1] convention.

    Integer dtypes use their full range, which is exact and unambiguous. Float
    inputs are already expected to be in [0, 1] per the KLA specification; a float
    image is only rescaled when it clearly uses 0-255 encoding *and* the caller
    permits that inference.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.max) if info.max > 1 else 1.0
    if assume_8bit_floats and sample_max > 2.0:
        return 255.0
    return 1.0


def load_image_float(
    path: str | os.PathLike[str],
    *,
    clip: bool = False,
    grayscale: bool | None = None,
    assume_8bit_floats: bool = True,
) -> LoadedImage:
    """Load an image as ``float32`` in the [0, 1] convention.

    Parameters
    ----------
    path:
        Image file. ``.npy`` is supported for float arrays saved by numpy.
    clip:
        Clip to [0, 1]. Use ``True`` for GT, ``False`` for NoisyLR -- the KLA
        specification states NoisyLR may legitimately exceed [0, 1].
    grayscale:
        ``None`` keeps the native channel count. ``True`` converts RGB to
        luminance. ``False`` is identical to ``None`` and exists so callers can
        express intent explicitly.
    assume_8bit_floats:
        Allow rescaling of float files whose maximum exceeds 2.0 by 255.

    Returns
    -------
    LoadedImage
        Always shaped ``(H, W, C)``.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        with Image.open(path) as handle:
            if handle.mode in {"I;16", "I;16B", "I;16L", "I"}:
                handle = handle.convert("I")
            elif handle.mode == "P":
                handle = handle.convert("RGB")
            elif handle.mode in {"LA", "RGBA"}:
                handle = handle.convert("L" if handle.mode == "LA" else "RGB")
            raw = np.asarray(handle)

    source_dtype = str(raw.dtype)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    elif raw.ndim == 3 and raw.shape[2] == 2:
        raw = raw[:, :, :1]
    elif raw.ndim != 3:
        raise ValueError(f"unsupported image shape {raw.shape} for {path}")

    sample_max = float(np.max(np.abs(raw))) if raw.size else 0.0
    divisor = _dtype_divisor(raw.dtype, sample_max, assume_8bit_floats)
    arr = raw.astype(np.float32, copy=True)
    if divisor != 1.0:
        arr /= np.float32(divisor)

    if grayscale and arr.shape[2] >= 3:
        arr = (arr[:, :, :3] * _BT601).sum(axis=2, keepdims=True).astype(np.float32)

    if clip:
        np.clip(arr, 0.0, 1.0, out=arr)

    return LoadedImage(
        array=np.ascontiguousarray(arr),
        source_dtype=source_dtype,
        scale_divisor=float(divisor),
        clipped=bool(clip),
        path=str(path),
    )


def save_image_float(
    path: str | os.PathLike[str],
    array: np.ndarray,
    *,
    bit_depth: int = 8,
    clip: bool = True,
) -> Path:
    """Save a float array in the [0, 1] convention as PNG/TIFF.

    KLA scores the images exactly as saved, so clipping and quantization happen
    here and nowhere else. Rounding is numpy's round-half-to-even.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    if bit_depth == 8:
        data = np.rint(arr * 255.0).astype(np.uint8)
    elif bit_depth == 16:
        data = np.rint(arr * 65535.0).astype(np.uint16)
    else:
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
    Image.fromarray(data).save(path)
    return path


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert ``(H, W, C)`` float array to a ``(C, H, W)`` float32 tensor."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ``(C, H, W)`` or ``(1, C, H, W)`` tensor to ``(H, W, C)`` array."""
    t = tensor.detach()
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"expected batch of 1, got {t.shape[0]}")
        t = t[0]
    return np.ascontiguousarray(t.float().cpu().numpy().transpose(1, 2, 0))


def image_files(folder: str | os.PathLike[str], exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    """Recursively discover image files under ``folder``, sorted for determinism."""
    root = Path(folder)
    if not root.exists():
        return []
    allowed = {e.lower() for e in exts}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


# --------------------------------------------------------------------------------------
# config / provenance
# --------------------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config. PyYAML is a training/eval dependency only."""
    import yaml  # local import keeps inference.py free of the dependency

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return p


def append_csv_row(path: str | os.PathLike[str], row: dict[str, Any], columns: Sequence[str] | None = None) -> Path:
    """Append a row to a CSV, writing the header when the file is new.

    Uses the stdlib ``csv`` module so training does not require pandas.
    """
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else list(row)
    exists = p.exists() and p.stat().st_size > 0
    with p.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in cols})
    return p


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a module."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def environment_snapshot() -> dict[str, Any]:
    """Collect a reproducibility snapshot of the current environment."""
    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        from PIL import __version__ as pil_version

        snap["pillow"] = pil_version
    except Exception:  # pragma: no cover
        snap["pillow"] = None
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        try:
            props = torch.cuda.get_device_properties(0)
            snap["gpu_name"] = props.name
            snap["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            snap["gpu_capability"] = f"{props.major}.{props.minor}"
        except Exception as exc:
            snap["gpu_error"] = str(exc)
    else:
        snap["gpu_name"] = None
    return snap


def human_bytes(num: float) -> str:
    """Format a byte count for logs."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"
