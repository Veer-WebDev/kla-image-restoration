#!/usr/bin/env python3
"""Evaluation harness: score the restoration model against the bicubic baseline.

This is the measurement instrument for the whole project. Every number that ends
up in a report, the experiment ledger or the presentation is produced here, from
a real run, and written to disk next to the inputs that produced it.

What it reports
---------------
* **Quality**: PSNR, SSIM, LPIPS, MAE, RMSE -- the KLA metric set -- for the model
  *and* for bicubic upsampling of the same input, per image and aggregated.
* **Cost**: parameter count, per-image latency (mean / median / p95 / best),
  throughput, and peak memory (CUDA allocator on GPU, peak process working set on
  CPU).
* **Robustness**: the same metrics grouped by degradation order, downsampling
  scale, downsampling kernel and noise-strength bucket, so a weakness cannot hide
  inside an average.
* **Explainability**: residual, absolute-error and improvement maps for the best,
  median and worst images. No classification metrics, no confusion matrices and no
  Grad-CAM appear anywhere: the task is restoration.

Modes
-----
``model`` (default)
    Load a checkpoint, rebuild the frozen evaluation split and run the model
    through exactly the same code path as ``inference.py`` (``restore_tensor``),
    including optional tiling. This is what measures latency.

``pred-dir`` (``--pred-dir``)
    Score images that are already on disk -- typically the output directory of a
    finished ``inference.py`` run -- against ground truth. No model is loaded, so
    no latency or parameter numbers are reported. This is the closest possible
    reproduction of how the organizers will score the submission.

Metric domain
-------------
``--metric-domain saved`` (default) quantizes predictions to the output bit depth
before scoring, because the organizers score the PNG files as written, not the
float tensors. ``--metric-domain float`` skips quantization and is directly
comparable with the validation numbers printed during training. The chosen domain
is recorded in every output file; the baseline is always treated identically.

Examples
--------
    python evaluate.py --checkpoint runs/baseline_residual_unet/best.pth
    python evaluate.py --checkpoint runs/exp/best.pth --split test --num-maps 8
    python evaluate.py --pred-dir runs/submission --gt-dir data/GT
    python evaluate.py --checkpoint runs/exp/best.pth --no-lpips --max-images 20
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from inference import restore_tensor  # noqa: E402  (same directory, shared restore path)
from kla_restore.checkpoint import load_model  # noqa: E402
from kla_restore.dataset import (  # noqa: E402
    GT_TOKENS,
    NOISY_TOKENS,
    DatasetConfig,
    RestorationDataset,
    build_dataloader,
    canonical_stem,
    detect_channels,
    discover_pairs,
    split_keys,
)
from kla_restore.degradation import DegradationConfig, describe_config  # noqa: E402
from kla_restore.metrics import aggregate, compute_metrics, get_lpips  # noqa: E402
from kla_restore.model import model_summary  # noqa: E402
from kla_restore.train import CSV_COLUMNS, load_config  # noqa: E402
from kla_restore.utils import (  # noqa: E402
    append_csv_row,
    environment_snapshot,
    get_logger,
    human_bytes,
    load_image_float,
    load_yaml,
    save_image_float,
    seed_everything,
    select_device,
    setup_logging,
    to_numpy,
    to_tensor,
    write_json,
)

LOGGER = get_logger()

#: Metrics reported per image, in report order.
METRIC_KEYS: tuple[str, ...] = ("psnr", "ssim", "lpips", "mae", "rmse")

#: Per-image CSV schema. Fixed so downstream tooling can rely on it.
PER_IMAGE_COLUMNS: tuple[str, ...] = (
    "key",
    "source",
    "order",
    "scale",
    "kernel",
    "gaussian_sigma",
    "speckle_sigma",
    "input_h",
    "input_w",
    "output_h",
    "output_w",
    "psnr",
    "ssim",
    "lpips",
    "mae",
    "rmse",
    "bicubic_psnr",
    "bicubic_ssim",
    "bicubic_lpips",
    "bicubic_mae",
    "bicubic_rmse",
    "psnr_delta",
    "ssim_delta",
    "lpips_delta",
    "mae_delta",
    "ms",
    "ms_best",
    "bicubic_ms",
    "megapixels",
    "note",
)

#: Robustness CSV schema (one row per group).
ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "factor",
    "group",
    "n",
    "psnr_mean",
    "psnr_std",
    "psnr_min",
    "bicubic_psnr_mean",
    "psnr_delta_mean",
    "ssim_mean",
    "bicubic_ssim_mean",
    "ssim_delta_mean",
    "lpips_mean",
    "mae_mean",
    "win_rate",
)

#: Evaluation ledger schema, richer than the shared training ledger.
EVAL_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "experiment_id",
    "eval_id",
    "mode",
    "split",
    "metric_domain",
    "checkpoint",
    "epoch",
    "eval_dir",
    "seed",
    "device",
    "gpu_name",
    "threads",
    "n_images",
    "channels",
    "eval_mode",
    "tile_size",
    "params_total",
    "params_millions",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "lpips_mean",
    "mae_mean",
    "rmse_mean",
    "bicubic_psnr_mean",
    "bicubic_ssim_mean",
    "bicubic_lpips_mean",
    "bicubic_mae_mean",
    "psnr_delta_mean",
    "psnr_delta_median",
    "ssim_delta_mean",
    "win_rate_psnr",
    "win_rate_ssim",
    "worst_key",
    "worst_psnr",
    "worst_delta_key",
    "worst_psnr_delta",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_best_ms",
    "megapixels_per_second",
    "bicubic_mean_ms",
    "peak_memory",
    "eval_seconds",
    "lpips_available",
    "torch_version",
    "notes",
)

#: Written next to the maps so the encodings are never guessed at.
MAP_LEGEND: dict[str, str] = {
    "input": "NoisyLR input, clipped to [0, 1] for display only.",
    "gt": "Ground truth, as loaded.",
    "bicubic": "Bicubic upsampling of the input to GT size (the baseline).",
    "pred": "Model restoration, exactly as inference.py would save it.",
    "residual": (
        "What the model added over bicubic: 0.5 + (pred - bicubic) * gain * 0.5, "
        "clipped to [0, 1]. Mid-grey is zero, brighter means the model raised the "
        "pixel above the baseline."
    ),
    "abserr": (
        "Absolute model error: |pred - gt| * gain, clipped to [0, 1]. Black is a "
        "perfect pixel, bright is a large error."
    ),
    "abserr_bicubic": "Absolute baseline error: |bicubic - gt| * gain, clipped to [0, 1].",
    "improvement": (
        "Where the model beats the baseline: 0.5 + (|bicubic - gt| - |pred - gt|) "
        "* gain * 0.5, clipped to [0, 1]. Brighter than mid-grey means the model is "
        "closer to GT than bicubic; darker means it is worse."
    ),
}


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------
def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or ``None`` if unavailable.

    Uses only the standard library: ``GetProcessMemoryInfo`` on Windows and
    ``getrusage`` elsewhere. Reported instead of a CUDA figure when evaluating on
    CPU, so the memory column is always a measurement and never an estimate.
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            # argtypes are mandatory here: the GetCurrentProcess pseudo-handle is
            # (HANDLE)-1, and without an explicit HANDLE argtype ctypes passes it
            # as a 32-bit int on 64-bit Windows, so the call fails with ok == 0.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE

            # psapi.dll forwards to kernel32 on modern Windows; try both so the
            # measurement survives on stripped-down images.
            query = None
            for dll_name, func_name in (
                ("psapi", "GetProcessMemoryInfo"),
                ("kernel32", "K32GetProcessMemoryInfo"),
            ):
                try:
                    dll = kernel32 if dll_name == "kernel32" else ctypes.WinDLL(dll_name, use_last_error=True)
                    query = getattr(dll, func_name)
                except (OSError, AttributeError):
                    continue
                query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
                query.restype = wintypes.BOOL
                break
            if query is None:
                LOGGER.debug("peak RSS unavailable: no GetProcessMemoryInfo export")
                return None

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if not query(get_current_process(), ctypes.byref(counters), counters.cb):
                LOGGER.debug(
                    "peak RSS unavailable: GetProcessMemoryInfo failed (error %d)",
                    ctypes.get_last_error(),
                )
                return None
            return int(counters.PeakWorkingSetSize)
        except Exception as exc:  # pragma: no cover - platform dependent
            LOGGER.debug("peak RSS unavailable: %s", exc)
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports kibibytes, macOS reports bytes.
        return value * 1024 if sys.platform.startswith("linux") else value
    except Exception as exc:  # pragma: no cover - platform dependent
        LOGGER.debug("peak RSS unavailable: %s", exc)
        return None


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for strict JSON.

    ``json.dump`` happily writes bare ``NaN``/``Infinity`` tokens, which no
    strict JSON parser accepts -- and the summary is meant to be machine-readable
    by graders and by our own report tooling. A missing metric becomes ``null``,
    which every parser understands, while numpy scalars are demoted to Python
    types so the file never depends on numpy to be read back.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def quantize(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """Round-trip a float array through the saved integer representation.

    Mirrors :func:`kla_restore.utils.save_image_float` exactly -- clip, then
    round-half-to-even onto the integer grid -- so ``--metric-domain saved``
    scores precisely the pixels that would land in the submitted PNG.
    """
    levels = {8: 255.0, 16: 65535.0}.get(int(bit_depth))
    if levels is None:
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
    clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return (np.rint(clipped * levels) / levels).astype(np.float32)


def bicubic_upsample(noisy: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """The mandated baseline: plain bicubic interpolation to the target size.

    Computed explicitly rather than read out of the model, so the baseline stays
    bicubic even if ``model.upsample_mode`` is ever changed by an ablation.
    """
    return F.interpolate(
        noisy,
        size=(int(size[0]), int(size[1])),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )


def signed_map(values: np.ndarray, gain: float) -> np.ndarray:
    """Encode a signed array around mid-grey for display."""
    scaled = 0.5 + np.asarray(values, dtype=np.float32) * float(gain) * 0.5
    return np.clip(scaled, 0.0, 1.0)


def magnitude_map(values: np.ndarray, gain: float) -> np.ndarray:
    """Encode a non-negative array with a visibility gain."""
    scaled = np.abs(np.asarray(values, dtype=np.float32)) * float(gain)
    return np.clip(scaled, 0.0, 1.0)


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile of finite values, or NaN when there is nothing to report."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q))


def mean_of(values: Sequence[float]) -> float:
    """Mean of finite values, or NaN."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def win_rate(deltas: Sequence[float]) -> float:
    """Fraction of images where the delta is strictly positive."""
    finite = [float(v) for v in deltas if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(1 for v in finite if v > 0.0) / len(finite))


def bucket_label(value: float, lo: float, hi: float, name: str) -> str:
    """Label a noise sigma as low/mid/high within its configured range."""
    if value is None or not math.isfinite(float(value)):
        return f"{name}_unknown"
    if float(value) <= 0.0:
        return f"{name}_off"
    if hi <= lo:
        return f"{name}_fixed"
    position = (float(value) - lo) / (hi - lo)
    if position < 1.0 / 3.0:
        return f"{name}_low"
    if position < 2.0 / 3.0:
        return f"{name}_mid"
    return f"{name}_high"


def iter_pairs(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one dict per sample, handling stacked and ragged batches alike.

    Evaluation runs full-size images whose shapes differ, so
    :func:`kla_restore.dataset.collate_variable` may hand back lists instead of
    tensors. Both shapes are unpacked here rather than assumed.
    """
    # Indexing a stacked (B, C, H, W) tensor and indexing a list of (C, H, W)
    # tensors both yield one (C, H, W) tensor, so a single path covers the two
    # collate shapes; ``[None]`` restores the batch axis the model expects.
    for index in range(len(batch["keys"])):
        yield {
            "noisy": batch["noisy"][index][None],
            "gt": batch["gt"][index][None],
            "key": batch["keys"][index],
            "params": batch["params"][index],
            "source": batch["sources"][index],
            "seed": batch["seeds"][index],
        }


# --------------------------------------------------------------------------------------
# split and dataset construction
# --------------------------------------------------------------------------------------
def resolve_split_keys(
    gt_map: dict[str, Path],
    *,
    split: str,
    split_file: str | None,
    ratios: Sequence[float],
    split_seed: int,
) -> tuple[list[str], str]:
    """Return the keys to evaluate and a human-readable provenance string.

    A ``split.json`` written by training is preferred, because it is the frozen
    split that produced the checkpoint. Recomputation from ratios and seed is the
    deterministic fallback, and any key missing from ``gt_map`` is reported rather
    than silently dropped.
    """
    if split_file:
        path = Path(split_file)
        if not path.exists():
            raise FileNotFoundError(f"split file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"split file must contain an object of key lists: {path}")
        if split == "all":
            keys = [k for name in ("train", "val", "test") for k in payload.get(name, [])]
        else:
            if split not in payload:
                raise KeyError(f"split {split!r} not present in {path}; found {sorted(payload)}")
            keys = list(payload[split])
        provenance = f"{path} [{split}]"
    else:
        splits = split_keys(gt_map.keys(), tuple(float(r) for r in ratios), int(split_seed))
        if split == "all":
            keys = [k for name in ("train", "val", "test") for k in splits[name]]
        else:
            if split not in splits:
                raise KeyError(f"unknown split {split!r}; expected train, val, test or all")
            keys = list(splits[split])
        provenance = f"recomputed ratios={tuple(ratios)} seed={split_seed} [{split}]"

    known = [k for k in keys if k in gt_map]
    unknown = [k for k in keys if k not in gt_map]
    if unknown:
        LOGGER.warning(
            "%d key(s) from the split have no GT file and were skipped: %s",
            len(unknown),
            unknown[:5],
        )
    if not known:
        raise ValueError(f"split {split!r} is empty after matching against {len(gt_map)} GT files")
    return sorted(known), provenance


def build_eval_dataset(
    keys: Sequence[str],
    gt_map: dict[str, Path],
    noisy_map: dict[str, Path],
    *,
    degradation: DegradationConfig,
    channels: int,
    eval_mode: str,
    max_eval_size: int | None,
    seed: int,
) -> RestorationDataset:
    """Build the frozen evaluation dataset (``training=False`` pins epoch 0)."""
    return RestorationDataset(
        list(keys),
        gt_map,
        noisy_map,
        mode=eval_mode,  # type: ignore[arg-type]
        degradation=degradation,
        config=DatasetConfig(
            patch_size=256,  # unused when training=False; full images are evaluated
            samples_per_image=1,
            synthetic_prob=0.0,
            augment=False,
            channels=channels,
            training=False,
            max_eval_size=max_eval_size,
            cache_images=False,
        ),
        seed=int(seed),
        # Evaluation must never be padded with substitutes behind our back.
        on_missing="error" if eval_mode == "official" else "synthetic",
    )


# --------------------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------------------
def score_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    baseline: np.ndarray | None,
    *,
    lpips_model: Any | None,
    device: torch.device,
) -> dict[str, float]:
    """Metrics for one prediction, plus the baseline and the deltas."""
    row: dict[str, float] = {}
    model_metrics = compute_metrics(pred, gt, lpips_model=lpips_model, device=device)
    for name in METRIC_KEYS:
        row[name] = float(model_metrics.get(name, float("nan")))
    if baseline is not None:
        base_metrics = compute_metrics(baseline, gt, lpips_model=lpips_model, device=device)
        for name in METRIC_KEYS:
            row[f"bicubic_{name}"] = float(base_metrics.get(name, float("nan")))
        row["psnr_delta"] = row["psnr"] - row["bicubic_psnr"]
        row["ssim_delta"] = row["ssim"] - row["bicubic_ssim"]
        row["mae_delta"] = row["bicubic_mae"] - row["mae"]  # positive = model better
        row["lpips_delta"] = row["bicubic_lpips"] - row["lpips"]  # positive = model better
    return row


def evaluate_model(
    model: torch.nn.Module,
    dataset: RestorationDataset,
    device: torch.device,
    *,
    lpips_model: Any | None,
    bit_depth: int,
    metric_domain: str,
    tile_size: int,
    tile_overlap: int,
    warmup: int,
    repeat: int,
    max_images: int,
    save_restored: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the model over the dataset and collect per-image records.

    The timed forward is :func:`inference.restore_tensor`, the very function the
    submitted entry point calls, so the reported latency describes the deliverable
    rather than a research-only path.
    """
    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        seed=int(dataset.seed),
        pin_memory=False,
    )
    records: list[dict[str, Any]] = []
    failures = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    total = min(len(dataset), max_images) if max_images > 0 else len(dataset)
    for batch in loader:
        for item in iter_pairs(batch):
            if len(records) + failures >= total:
                break
            key = str(item["key"])
            try:
                noisy = item["noisy"].to(device)
                gt_tensor = item["gt"]
                target_size = (int(gt_tensor.shape[-2]), int(gt_tensor.shape[-1]))

                for _ in range(max(0, warmup)):
                    restore_tensor(
                        model, noisy, target_size, tile_size=tile_size, tile_overlap=tile_overlap
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                timings: list[float] = []
                restored = None
                for _ in range(max(1, repeat)):
                    mark = time.perf_counter()
                    restored = restore_tensor(
                        model, noisy, target_size, tile_size=tile_size, tile_overlap=tile_overlap
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings.append((time.perf_counter() - mark) * 1000.0)

                mark = time.perf_counter()
                baseline_tensor = bicubic_upsample(noisy, target_size)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                bicubic_ms = (time.perf_counter() - mark) * 1000.0

                pred_np = to_numpy(restored[0].float().cpu())  # type: ignore[index]
                base_np = np.clip(to_numpy(baseline_tensor[0].float().cpu()), 0.0, 1.0)
                gt_np = to_numpy(gt_tensor[0].float())
                if metric_domain == "saved":
                    pred_np = quantize(pred_np, bit_depth)
                    base_np = quantize(base_np, bit_depth)

                if save_restored is not None:
                    save_image_float(
                        save_restored / f"{key}.png", pred_np, bit_depth=bit_depth
                    )

                row: dict[str, Any] = {
                    "key": key,
                    "source": str(item["source"]),
                    "input_h": int(noisy.shape[-2]),
                    "input_w": int(noisy.shape[-1]),
                    "output_h": target_size[0],
                    "output_w": target_size[1],
                    "ms": float(np.mean(timings)),
                    "ms_best": float(np.min(timings)),
                    "bicubic_ms": float(bicubic_ms),
                    "megapixels": target_size[0] * target_size[1] / 1e6,
                    "note": "",
                }
                params = item["params"] if isinstance(item["params"], dict) else {}
                row["order"] = str(params.get("order", "unknown"))
                row["scale"] = params.get("scale", 0)
                row["kernel"] = str(params.get("kernel", "unknown"))
                row["gaussian_sigma"] = params.get("gaussian_sigma", float("nan"))
                row["speckle_sigma"] = params.get("speckle_sigma", float("nan"))
                row.update(
                    score_pair(
                        pred_np, gt_np, base_np, lpips_model=lpips_model, device=device
                    )
                )
                records.append(row)
                LOGGER.info(
                    "[%d/%d] %s %dx%d -> %dx%d | psnr %.3f (bicubic %.3f, %+.3f) | ssim %.4f | %.1f ms",
                    len(records),
                    total,
                    key,
                    row["input_h"],
                    row["input_w"],
                    row["output_h"],
                    row["output_w"],
                    row["psnr"],
                    row["bicubic_psnr"],
                    row["psnr_delta"],
                    row["ssim"],
                    row["ms"],
                )
            except Exception as exc:
                failures += 1
                LOGGER.error("failed on %s: %s", key, exc)
        if len(records) + failures >= total:
            break

    elapsed = time.perf_counter() - started
    timing = {
        "eval_seconds": elapsed,
        "failures": failures,
        "attempted": len(records) + failures,
    }
    return records, timing


def evaluate_pred_dir(
    pred_dir: Path,
    gt_dir: Path,
    noisy_dir: Path | None,
    *,
    lpips_model: Any | None,
    device: torch.device,
    channels: int,
    bit_depth: int,
    metric_domain: str,
    keys_filter: Sequence[str] | None,
    max_images: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score restored images already on disk against ground truth.

    Size mismatches are recorded as failures instead of being resized away: a
    resize here would invent agreement that the submitted files do not have.
    """
    gt_map, pred_map, report = discover_pairs(gt_dir, pred_dir)
    if not pred_map:
        raise FileNotFoundError(f"no readable images in {pred_dir}")
    report.log()
    noisy_map: dict[str, Path] = {}
    if noisy_dir is not None:
        _, noisy_map, _ = discover_pairs(gt_dir, noisy_dir)

    keys = sorted(set(gt_map) & set(pred_map))
    if keys_filter is not None:
        allowed = set(keys_filter)
        keys = [k for k in keys if k in allowed]
    if not keys:
        raise ValueError(
            f"no GT/prediction pairs matched. GT keys: {len(gt_map)}, prediction keys: "
            f"{len(pred_map)}. Check the filenames against the pairing tokens "
            f"{NOISY_TOKENS + GT_TOKENS}."
        )
    if max_images > 0:
        keys = keys[:max_images]

    records: list[dict[str, Any]] = []
    failures = 0
    started = time.perf_counter()
    for index, key in enumerate(keys, start=1):
        try:
            gt_loaded = load_image_float(gt_map[key], clip=True, grayscale=channels == 1)
            pred_loaded = load_image_float(pred_map[key], clip=True, grayscale=channels == 1)
            gt_np = gt_loaded.array
            pred_np = pred_loaded.array
            if channels == 3 and pred_np.shape[2] == 1:
                pred_np = np.repeat(pred_np, 3, axis=2)
            if channels == 3 and gt_np.shape[2] == 1:
                gt_np = np.repeat(gt_np, 3, axis=2)
            if pred_np.shape != gt_np.shape:
                failures += 1
                LOGGER.error(
                    "size mismatch for %s: prediction %s vs GT %s (not resized)",
                    key,
                    pred_np.shape,
                    gt_np.shape,
                )
                records.append(
                    {
                        "key": key,
                        "source": "pred_dir",
                        "order": "unknown",
                        "scale": 0,
                        "kernel": "unknown",
                        "gaussian_sigma": float("nan"),
                        "speckle_sigma": float("nan"),
                        "input_h": 0,
                        "input_w": 0,
                        "output_h": int(pred_np.shape[0]),
                        "output_w": int(pred_np.shape[1]),
                        "megapixels": pred_np.shape[0] * pred_np.shape[1] / 1e6,
                        "note": f"size mismatch pred={pred_np.shape[:2]} gt={gt_np.shape[:2]}",
                    }
                )
                continue

            base_np = None
            input_hw = (0, 0)
            if key in noisy_map:
                noisy_loaded = load_image_float(
                    noisy_map[key], clip=False, grayscale=channels == 1
                )
                noisy_np = noisy_loaded.array
                if channels == 3 and noisy_np.shape[2] == 1:
                    noisy_np = np.repeat(noisy_np, 3, axis=2)
                input_hw = noisy_loaded.hw
                tensor = to_tensor(noisy_np)[None]
                base_np = np.clip(
                    to_numpy(
                        bicubic_upsample(tensor, (gt_np.shape[0], gt_np.shape[1]))[0]
                    ),
                    0.0,
                    1.0,
                )
                if metric_domain == "saved":
                    base_np = quantize(base_np, bit_depth)

            if metric_domain == "saved":
                pred_np = quantize(pred_np, bit_depth)

            row: dict[str, Any] = {
                "key": key,
                "source": "pred_dir",
                "order": "unknown",
                "scale": 0,
                "kernel": "unknown",
                "gaussian_sigma": float("nan"),
                "speckle_sigma": float("nan"),
                "input_h": int(input_hw[0]),
                "input_w": int(input_hw[1]),
                "output_h": int(gt_np.shape[0]),
                "output_w": int(gt_np.shape[1]),
                "megapixels": gt_np.shape[0] * gt_np.shape[1] / 1e6,
                "note": "",
            }
            row.update(
                score_pair(pred_np, gt_np, base_np, lpips_model=lpips_model, device=device)
            )
            records.append(row)
            LOGGER.info(
                "[%d/%d] %s %dx%d | psnr %.3f | ssim %.4f%s",
                index,
                len(keys),
                key,
                row["output_h"],
                row["output_w"],
                row["psnr"],
                row["ssim"],
                (
                    f" | bicubic {row['bicubic_psnr']:.3f} ({row['psnr_delta']:+.3f})"
                    if base_np is not None
                    else ""
                ),
            )
        except Exception as exc:
            failures += 1
            LOGGER.error("failed on %s: %s", key, exc)

    timing = {
        "eval_seconds": time.perf_counter() - started,
        "failures": failures,
        "attempted": len(keys),
    }
    return records, timing


# --------------------------------------------------------------------------------------
# aggregation, robustness, maps
# --------------------------------------------------------------------------------------
def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image records into the headline numbers."""
    scored = [r for r in records if "psnr" in r]
    if not scored:
        return {"n": 0}
    model_rows = [{k: float(r[k]) for k in METRIC_KEYS if k in r} for r in scored]
    out = aggregate(model_rows, METRIC_KEYS)
    has_baseline = any("bicubic_psnr" in r for r in scored)
    if has_baseline:
        base_rows = [
            {k: float(r[f"bicubic_{k}"]) for k in METRIC_KEYS if f"bicubic_{k}" in r}
            for r in scored
        ]
        for key, value in aggregate(base_rows, METRIC_KEYS).items():
            out[f"bicubic_{key}"] = value
        psnr_deltas = [r.get("psnr_delta", float("nan")) for r in scored]
        ssim_deltas = [r.get("ssim_delta", float("nan")) for r in scored]
        out["psnr_delta_mean"] = mean_of(psnr_deltas)
        out["psnr_delta_median"] = percentile(psnr_deltas, 50)
        out["psnr_delta_min"] = percentile(psnr_deltas, 0)
        out["psnr_delta_max"] = percentile(psnr_deltas, 100)
        out["ssim_delta_mean"] = mean_of(ssim_deltas)
        out["mae_delta_mean"] = mean_of([r.get("mae_delta", float("nan")) for r in scored])
        out["lpips_delta_mean"] = mean_of([r.get("lpips_delta", float("nan")) for r in scored])
        out["win_rate_psnr"] = win_rate(psnr_deltas)
        out["win_rate_ssim"] = win_rate(ssim_deltas)
    return out


def latency_stats(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Latency and throughput, from the timed forwards only."""
    times = [float(r["ms"]) for r in records if "ms" in r and math.isfinite(float(r["ms"]))]
    if not times:
        return {}
    megapixels = [float(r.get("megapixels", 0.0)) for r in records if "ms" in r]
    total_mp = float(sum(megapixels))
    total_s = float(sum(times)) / 1000.0
    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_median_ms": float(statistics.median(times)),
        "latency_p95_ms": percentile(times, 95),
        "latency_min_ms": float(np.min(times)),
        "latency_max_ms": float(np.max(times)),
        "latency_best_ms": float(
            np.mean([float(r["ms_best"]) for r in records if "ms_best" in r])
        ),
        "bicubic_mean_ms": mean_of([r.get("bicubic_ms", float("nan")) for r in records]),
        "megapixels_per_second": (total_mp / total_s) if total_s > 0 else float("nan"),
        "images_per_second": (len(times) / total_s) if total_s > 0 else float("nan"),
    }


def robustness_groups(
    records: Sequence[dict[str, Any]], degradation: DegradationConfig
) -> list[dict[str, Any]]:
    """Group metrics by degradation factor so weaknesses cannot hide in a mean."""
    scored = [r for r in records if "psnr" in r]
    if not scored:
        return []
    g_lo, g_hi = degradation.gaussian_sigma
    s_lo, s_hi = degradation.speckle_sigma

    factors: dict[str, dict[str, list[dict[str, Any]]]] = {
        "order": {},
        "scale": {},
        "kernel": {},
        "gaussian": {},
        "speckle": {},
        "source": {},
    }
    for row in scored:
        labels = {
            "order": str(row.get("order", "unknown")),
            "scale": f"x{row.get('scale', 0)}",
            "kernel": str(row.get("kernel", "unknown")),
            "gaussian": bucket_label(
                float(row.get("gaussian_sigma", float("nan"))), g_lo, g_hi, "gaussian"
            ),
            "speckle": bucket_label(
                float(row.get("speckle_sigma", float("nan"))), s_lo, s_hi, "speckle"
            ),
            "source": str(row.get("source", "unknown")),
        }
        for factor, label in labels.items():
            factors[factor].setdefault(label, []).append(row)

    out: list[dict[str, Any]] = []
    for factor, groups in factors.items():
        if len(groups) <= 1 and factor in {"order", "scale", "kernel", "source"}:
            # A single level carries no comparative information; keep it only when
            # it is a noise bucket, where the level itself is informative.
            if factor != "source":
                continue
        for label in sorted(groups):
            rows = groups[label]
            out.append(
                {
                    "factor": factor,
                    "group": label,
                    "n": len(rows),
                    "psnr_mean": mean_of([r["psnr"] for r in rows]),
                    "psnr_std": (
                        float(np.std([r["psnr"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
                    ),
                    "psnr_min": percentile([r["psnr"] for r in rows], 0),
                    "bicubic_psnr_mean": mean_of(
                        [r.get("bicubic_psnr", float("nan")) for r in rows]
                    ),
                    "psnr_delta_mean": mean_of([r.get("psnr_delta", float("nan")) for r in rows]),
                    "ssim_mean": mean_of([r["ssim"] for r in rows]),
                    "bicubic_ssim_mean": mean_of(
                        [r.get("bicubic_ssim", float("nan")) for r in rows]
                    ),
                    "ssim_delta_mean": mean_of([r.get("ssim_delta", float("nan")) for r in rows]),
                    "lpips_mean": mean_of([r.get("lpips", float("nan")) for r in rows]),
                    "mae_mean": mean_of([r.get("mae", float("nan")) for r in rows]),
                    "win_rate": win_rate([r.get("psnr_delta", float("nan")) for r in rows]),
                }
            )
    return out


def select_map_keys(records: Sequence[dict[str, Any]], count: int) -> list[str]:
    """Pick which images get qualitative maps.

    Chooses the worst, median and best by improvement over bicubic, plus the
    single lowest absolute PSNR, then fills the remainder at evenly spaced ranks.
    The worst cases are selected *first* so an honest failure case is guaranteed to
    exist in the figures rather than being hunted for later.
    """
    scored = [r for r in records if "psnr" in r]
    if not scored or count <= 0:
        return []
    by_delta = sorted(scored, key=lambda r: float(r.get("psnr_delta", 0.0)))
    by_psnr = sorted(scored, key=lambda r: float(r["psnr"]))
    chosen: list[str] = []

    def push(row: dict[str, Any]) -> None:
        key = str(row["key"])
        if key not in chosen and len(chosen) < count:
            chosen.append(key)

    push(by_delta[0])  # worst improvement: the honest failure case
    push(by_psnr[0])  # hardest image in absolute terms
    push(by_delta[len(by_delta) // 2])  # typical
    push(by_delta[-1])  # best improvement
    if len(chosen) < count and len(by_delta) > 1:
        remaining = count - len(chosen)
        for step in range(remaining):
            index = int(round((step + 1) * (len(by_delta) - 1) / (remaining + 1)))
            push(by_delta[index])
    return chosen


def write_maps_for_model(
    model: torch.nn.Module,
    dataset: RestorationDataset,
    keys: Sequence[str],
    device: torch.device,
    maps_dir: Path,
    *,
    bit_depth: int,
    metric_domain: str,
    tile_size: int,
    tile_overlap: int,
    error_gain: float,
    residual_gain: float,
) -> list[str]:
    """Re-run the selected images and write the qualitative maps."""
    index_of = {key: position for position, key in enumerate(dataset.keys)}
    written: list[str] = []
    maps_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in index_of:
            LOGGER.warning("cannot write maps for %s: not in the evaluated dataset", key)
            continue
        try:
            sample = dataset[index_of[key]]
            noisy = sample["noisy"][None].to(device)
            gt_np = to_numpy(sample["gt"].float())
            target_size = (gt_np.shape[0], gt_np.shape[1])
            restored = restore_tensor(
                model, noisy, target_size, tile_size=tile_size, tile_overlap=tile_overlap
            )
            pred_np = to_numpy(restored[0].float().cpu())
            base_np = np.clip(
                to_numpy(bicubic_upsample(noisy, target_size)[0].float().cpu()), 0.0, 1.0
            )
            if metric_domain == "saved":
                pred_np = quantize(pred_np, bit_depth)
                base_np = quantize(base_np, bit_depth)

            noisy_np = np.clip(to_numpy(sample["noisy"].float()), 0.0, 1.0)
            abs_pred = np.abs(pred_np - gt_np)
            abs_base = np.abs(base_np - gt_np)
            outputs = {
                "input": noisy_np,
                "gt": gt_np,
                "bicubic": base_np,
                "pred": pred_np,
                "residual": signed_map(pred_np - base_np, residual_gain),
                "abserr": magnitude_map(abs_pred, error_gain),
                "abserr_bicubic": magnitude_map(abs_base, error_gain),
                "improvement": signed_map(abs_base - abs_pred, error_gain),
            }
            for name, array in outputs.items():
                save_image_float(maps_dir / f"{key}__{name}.png", array, bit_depth=8)
            written.append(key)
            LOGGER.info("maps written for %s", key)
        except Exception as exc:
            LOGGER.error("map generation failed for %s: %s", key, exc)
    return written


def write_maps_for_pred_dir(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    gt_dir: Path,
    pred_dir: Path,
    noisy_dir: Path | None,
    maps_dir: Path,
    *,
    channels: int,
    bit_depth: int,
    metric_domain: str,
    error_gain: float,
    residual_gain: float,
) -> list[str]:
    """Write qualitative maps from files on disk (``--pred-dir`` mode)."""
    gt_map, pred_map, _ = discover_pairs(gt_dir, pred_dir)
    noisy_map: dict[str, Path] = {}
    if noisy_dir is not None:
        _, noisy_map, _ = discover_pairs(gt_dir, noisy_dir)
    written: list[str] = []
    maps_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in gt_map or key not in pred_map:
            LOGGER.warning("cannot write maps for %s: missing GT or prediction", key)
            continue
        try:
            gt_np = load_image_float(gt_map[key], clip=True, grayscale=channels == 1).array
            pred_np = load_image_float(pred_map[key], clip=True, grayscale=channels == 1).array
            if pred_np.shape != gt_np.shape:
                LOGGER.warning("skipping maps for %s: size mismatch", key)
                continue
            if metric_domain == "saved":
                pred_np = quantize(pred_np, bit_depth)
            abs_pred = np.abs(pred_np - gt_np)
            outputs = {
                "gt": gt_np,
                "pred": pred_np,
                "abserr": magnitude_map(abs_pred, error_gain),
            }
            if key in noisy_map:
                noisy_loaded = load_image_float(
                    noisy_map[key], clip=False, grayscale=channels == 1
                )
                tensor = to_tensor(noisy_loaded.array)[None]
                base_np = np.clip(
                    to_numpy(bicubic_upsample(tensor, (gt_np.shape[0], gt_np.shape[1]))[0]),
                    0.0,
                    1.0,
                )
                if metric_domain == "saved":
                    base_np = quantize(base_np, bit_depth)
                abs_base = np.abs(base_np - gt_np)
                outputs["input"] = np.clip(noisy_loaded.array, 0.0, 1.0)
                outputs["bicubic"] = base_np
                outputs["residual"] = signed_map(pred_np - base_np, residual_gain)
                outputs["abserr_bicubic"] = magnitude_map(abs_base, error_gain)
                outputs["improvement"] = signed_map(abs_base - abs_pred, error_gain)
            for name, array in outputs.items():
                save_image_float(maps_dir / f"{key}__{name}.png", array, bit_depth=8)
            written.append(key)
            LOGGER.info("maps written for %s", key)
        except Exception as exc:
            LOGGER.error("map generation failed for %s: %s", key, exc)
    return written


def failure_cases(records: Sequence[dict[str, Any]], count: int) -> dict[str, Any]:
    """The worst images by absolute quality and by improvement over bicubic.

    Reported unconditionally. Downsampling destroys high-frequency detail that no
    deterministic model can recover, so an honest submission shows where it loses.
    """
    scored = [r for r in records if "psnr" in r]
    if not scored:
        return {"worst_psnr": [], "worst_delta": []}
    fields = ("key", "psnr", "ssim", "bicubic_psnr", "psnr_delta", "order", "scale", "kernel")

    def slim(row: dict[str, Any]) -> dict[str, Any]:
        return {f: row.get(f) for f in fields if f in row}

    by_psnr = sorted(scored, key=lambda r: float(r["psnr"]))[:count]
    by_delta = sorted(scored, key=lambda r: float(r.get("psnr_delta", 0.0)))[:count]
    return {
        "worst_psnr": [slim(r) for r in by_psnr],
        "worst_delta": [slim(r) for r in by_delta],
        "n_regressions": sum(
            1 for r in scored if float(r.get("psnr_delta", 0.0)) < 0.0 and "psnr_delta" in r
        ),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Score the restoration model against the bicubic baseline: quality, cost, "
            "robustness and qualitative maps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config for data settings")
    parser.add_argument(
        "--degradation-config",
        default="configs/degradation.yaml",
        help="YAML degradation config used for synthetic evaluation",
    )
    parser.add_argument("--checkpoint", default=None, help="model weights (.pth); required in model mode")
    parser.add_argument("--gt-dir", default=None, help="ground-truth directory (default: config data.gt_dir)")
    parser.add_argument("--noisy-dir", default=None, help="official NoisyLR directory, if available")
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="score existing restored images instead of running the model",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test", "all"),
        help="which split to evaluate; test is for final reporting only",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="split.json written by training (preferred over recomputing the split)",
    )
    parser.add_argument(
        "--eval-mode",
        default="auto",
        choices=("auto", "official", "synthetic"),
        help="use official NoisyLR pairs or synthesize the degradation",
    )
    parser.add_argument("--output-dir", default=None, help="where to write results (default: alongside the checkpoint)")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--seed", type=int, default=None, help="override the evaluation seed")
    parser.add_argument(
        "--metric-domain",
        default="saved",
        choices=("saved", "float"),
        help="score the quantized saved pixels (default) or raw floats",
    )
    parser.add_argument("--bit-depth", type=int, default=None, choices=(8, 16), help="output bit depth")
    parser.add_argument("--channels", type=int, default=None, choices=(1, 3), help="force channel count")
    parser.add_argument("--tile-size", type=int, default=None, help="tile large inputs; 0 disables")
    parser.add_argument("--tile-overlap", type=int, default=None, help="tile overlap in pixels")
    parser.add_argument("--max-images", type=int, default=0, help="evaluate at most N images; 0 = all")
    parser.add_argument("--warmup", type=int, default=1, help="untimed forwards before timing each image")
    parser.add_argument("--repeat", type=int, default=1, help="timed forwards per image")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 keeps the default")
    parser.add_argument("--num-maps", type=int, default=6, help="how many images get qualitative maps")
    parser.add_argument("--no-maps", action="store_true", help="skip qualitative maps entirely")
    parser.add_argument("--error-gain", type=float, default=5.0, help="display gain for error maps")
    parser.add_argument("--residual-gain", type=float, default=5.0, help="display gain for residual maps")
    parser.add_argument("--failure-cases", type=int, default=5, help="how many worst cases to record")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS (never needed at inference)")
    parser.add_argument("--lpips-net", default="alex", help="LPIPS backbone")
    parser.add_argument("--save-restored", default=None, help="also write the model outputs here")
    parser.add_argument("--tag", default=None, help="suffix for the evaluation id and output directory")
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="do not append to results/experiments.csv and results/evaluations.csv",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--log-file", default=None, help="also write logs to this file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_file, level=args.log_level)
    wall_started = time.perf_counter()

    try:
        config = load_config(args.config, None)
    except Exception as exc:
        LOGGER.error("could not load config %s: %s", args.config, exc)
        return 2

    degradation_path = Path(args.degradation_config)
    degradation_dict: dict[str, Any] = {}
    if degradation_path.exists():
        try:
            degradation_dict = load_yaml(degradation_path)
        except Exception as exc:
            LOGGER.error("could not load degradation config %s: %s", degradation_path, exc)
            return 2
    elif args.degradation_config != "configs/degradation.yaml":
        LOGGER.error("degradation config not found: %s", degradation_path)
        return 2

    gt_dir = Path(args.gt_dir or config["data"]["gt_dir"])
    if not gt_dir.is_dir():
        LOGGER.error("GT directory not found: %s", gt_dir.resolve())
        return 2
    noisy_dir_arg = args.noisy_dir if args.noisy_dir is not None else config["data"].get("noisy_dir")
    noisy_dir = Path(noisy_dir_arg) if noisy_dir_arg else None
    if noisy_dir is not None and not noisy_dir.is_dir():
        LOGGER.error("NoisyLR directory not found: %s", noisy_dir.resolve())
        return 2

    seed = int(args.seed if args.seed is not None else config["seed"])
    seed_everything(seed, strict=bool(config.get("strict_determinism", False)))
    device = select_device(args.device)
    if int(args.threads) > 0:
        torch.set_num_threads(int(args.threads))
    if args.split == "test":
        LOGGER.warning(
            "evaluating the TEST split: report these numbers, never tune on them"
        )

    model = None
    meta = None
    params_summary: dict[str, Any] = {}
    if args.pred_dir is None:
        if not args.checkpoint:
            LOGGER.error("model mode needs --checkpoint (or use --pred-dir to score saved images)")
            return 2
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            LOGGER.error("checkpoint not found: %s", checkpoint_path)
            return 2
        try:
            model, meta = load_model(checkpoint_path, map_location=device)
        except Exception as exc:
            LOGGER.error("failed to load %s: %s", checkpoint_path, exc)
            return 2
        model = model.to(device).eval()
        params_summary = model_summary(model)
    else:
        checkpoint_path = None

    channels = int(
        args.channels
        if args.channels is not None
        else (meta.channels if meta is not None else config["data"]["channels"])
    )
    bit_depth = int(
        args.bit_depth
        if args.bit_depth is not None
        else (meta.bit_depth if meta is not None else config["inference"]["bit_depth"])
    )
    tile_size = int(args.tile_size if args.tile_size is not None else config["inference"]["tile_size"])
    tile_overlap = int(
        args.tile_overlap if args.tile_overlap is not None else config["inference"]["tile_overlap"]
    )

    try:
        degradation = DegradationConfig.from_dict(
            (meta.degradation_config if meta is not None and meta.degradation_config else degradation_dict)
        )
    except Exception as exc:
        LOGGER.error("invalid degradation config: %s", exc)
        return 2

    lpips_model = None
    lpips_available = False
    if not args.no_lpips:
        try:
            lpips_model = get_lpips(args.lpips_net, device)
            lpips_available = True
        except Exception as exc:
            LOGGER.warning("LPIPS unavailable, continuing without it: %s", exc)

    eval_id = args.tag or time.strftime("%Y%m%d_%H%M%S")
    experiment_id = str(meta.experiment_id) if meta is not None else "pred_dir"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif checkpoint_path is not None:
        output_dir = checkpoint_path.parent / f"eval_{args.split}_{eval_id}"
    else:
        output_dir = Path(config["paths"]["output_dir"]) / f"eval_preddir_{eval_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_restored = Path(args.save_restored) if args.save_restored else None
    if save_restored is not None:
        save_restored.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 78)
    LOGGER.info("mode         : %s", "pred-dir" if args.pred_dir else "model")
    if checkpoint_path is not None and meta is not None:
        LOGGER.info(
            "checkpoint   : %s (experiment %s, epoch %d)", checkpoint_path, meta.experiment_id, meta.epoch
        )
        LOGGER.info(
            "params       : %s (%.4f M)", params_summary["params_total"], params_summary["params_millions"]
        )
    LOGGER.info("gt dir       : %s", gt_dir)
    LOGGER.info("noisy dir    : %s", noisy_dir if noisy_dir else "none (synthetic degradation)")
    LOGGER.info("device       : %s (threads=%d)", device, torch.get_num_threads())
    LOGGER.info("metric domain: %s (bit depth %d)", args.metric_domain, bit_depth)
    LOGGER.info("lpips        : %s", "enabled" if lpips_available else "disabled")
    LOGGER.info("output dir   : %s", output_dir)
    LOGGER.info("=" * 78)

    dataset: RestorationDataset | None = None
    provenance = "n/a"
    eval_mode = args.eval_mode
    try:
        gt_map, noisy_map, pair_report = discover_pairs(gt_dir, noisy_dir)
        if not gt_map:
            LOGGER.error("no readable images in %s", gt_dir.resolve())
            return 2
        pair_report.log()
        if args.channels is None and meta is None and config["data"].get("autodetect_channels", True):
            channels = detect_channels(gt_map)
            LOGGER.info("channels detected from data: %d", channels)

        split_keys_list, provenance = resolve_split_keys(
            gt_map,
            split=args.split,
            split_file=args.split_file,
            ratios=config["data"]["split_ratios"],
            split_seed=int(config["data"]["split_seed"]),
        )
        LOGGER.info("split        : %d image(s) via %s", len(split_keys_list), provenance)

        if args.pred_dir is not None:
            records, timing = evaluate_pred_dir(
                Path(args.pred_dir),
                gt_dir,
                noisy_dir,
                lpips_model=lpips_model,
                device=device,
                channels=channels,
                bit_depth=bit_depth,
                metric_domain=args.metric_domain,
                keys_filter=None if args.split == "all" else split_keys_list,
                max_images=int(args.max_images),
            )
            eval_mode = "pred_dir"
        else:
            if eval_mode == "auto":
                eval_mode = "official" if noisy_map else "synthetic"
            if eval_mode == "official" and not noisy_map:
                LOGGER.error("--eval-mode official needs NoisyLR files; none were found")
                return 2
            dataset = build_eval_dataset(
                split_keys_list,
                gt_map,
                noisy_map,
                degradation=degradation,
                channels=channels,
                eval_mode=eval_mode,
                max_eval_size=config["data"].get("max_eval_size"),
                seed=seed,
            )
            assert model is not None  # guarded above
            records, timing = evaluate_model(
                model,
                dataset,
                device,
                lpips_model=lpips_model,
                bit_depth=bit_depth,
                metric_domain=args.metric_domain,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                warmup=int(args.warmup),
                repeat=int(args.repeat),
                max_images=int(args.max_images),
                save_restored=save_restored,
            )
    except Exception as exc:
        LOGGER.error("evaluation failed: %s", exc, exc_info=args.log_level.upper() == "DEBUG")
        return 1

    scored = [r for r in records if "psnr" in r]
    if not scored:
        LOGGER.error("no image was scored successfully; nothing to report")
        return 1

    summary_metrics = summarize(records)
    latency = latency_stats(records)
    robustness = robustness_groups(records, degradation)
    failures = failure_cases(records, int(args.failure_cases))

    map_keys: list[str] = []
    if not args.no_maps and int(args.num_maps) > 0:
        map_keys = select_map_keys(records, int(args.num_maps))
        maps_dir = output_dir / "maps"
        if dataset is not None and model is not None:
            written = write_maps_for_model(
                model,
                dataset,
                map_keys,
                device,
                maps_dir,
                bit_depth=bit_depth,
                metric_domain=args.metric_domain,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                error_gain=float(args.error_gain),
                residual_gain=float(args.residual_gain),
            )
        else:
            written = write_maps_for_pred_dir(
                records,
                map_keys,
                gt_dir,
                Path(args.pred_dir),  # type: ignore[arg-type]
                noisy_dir,
                maps_dir,
                channels=channels,
                bit_depth=bit_depth,
                metric_domain=args.metric_domain,
                error_gain=float(args.error_gain),
                residual_gain=float(args.residual_gain),
            )
        map_keys = written
        if written:
            write_json(
                maps_dir / "legend.json",
                json_safe(
                    {
                        "encodings": MAP_LEGEND,
                        "error_gain": float(args.error_gain),
                        "residual_gain": float(args.residual_gain),
                        "metric_domain": args.metric_domain,
                        "keys": written,
                        "selection": (
                            "worst improvement over bicubic, lowest absolute PSNR, median "
                            "improvement, best improvement, then evenly spaced ranks"
                        ),
                    }
                ),
            )

    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_rss = peak_rss_bytes()
    peak_memory = (
        human_bytes(peak_cuda)
        if peak_cuda is not None
        else (f"{human_bytes(peak_rss)} rss" if peak_rss is not None else "n/a")
    )

    for column in PER_IMAGE_COLUMNS:
        for row in records:
            row.setdefault(column, "")
    per_image_path = output_dir / "metrics_per_image.csv"
    if per_image_path.exists():
        per_image_path.unlink()
    for row in records:
        append_csv_row(per_image_path, row, PER_IMAGE_COLUMNS)

    robustness_path = output_dir / "robustness.csv"
    if robustness_path.exists():
        robustness_path.unlink()
    for row in robustness:
        append_csv_row(robustness_path, row, ROBUSTNESS_COLUMNS)

    summary = {
        "eval_id": eval_id,
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "pred_dir" if args.pred_dir else "model",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_epoch": int(meta.epoch) if meta is not None else None,
        "pred_dir": str(args.pred_dir) if args.pred_dir else None,
        "gt_dir": str(gt_dir),
        "noisy_dir": str(noisy_dir) if noisy_dir else None,
        "split": args.split,
        "split_provenance": provenance,
        "split_size": len(split_keys_list),
        "eval_mode": eval_mode,
        "metric_domain": args.metric_domain,
        "bit_depth": bit_depth,
        "channels": channels,
        "seed": seed,
        "device": str(device),
        "threads": int(torch.get_num_threads()),
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "n_scored": len(scored),
        "n_failed": int(timing.get("failures", 0)),
        "lpips_available": lpips_available,
        "lpips_net": args.lpips_net if lpips_available else None,
        "metrics": summary_metrics,
        "latency": latency,
        "params": params_summary or None,
        "peak_memory": peak_memory,
        "peak_memory_cuda_bytes": peak_cuda,
        "peak_rss_bytes": peak_rss,
        "eval_seconds": round(float(timing.get("eval_seconds", 0.0)), 2),
        "wall_seconds": round(time.perf_counter() - wall_started, 2),
        "degradation": describe_config(degradation),
        "degradation_config": degradation.to_dict(),
        "degradation_calibrated": bool(degradation.metadata.get("calibrated", False)),
        "robustness": robustness,
        "failure_cases": failures,
        "map_keys": map_keys,
        "environment": environment_snapshot(),
        "outputs": {
            "per_image_csv": str(per_image_path),
            "robustness_csv": str(robustness_path),
            "maps_dir": str(output_dir / "maps") if map_keys else None,
            "restored_dir": str(save_restored) if save_restored else None,
        },
    }
    summary_path = write_json(output_dir / "summary.json", json_safe(summary))

    if not args.no_ledger:
        eval_row = {
            "timestamp": summary["timestamp"],
            "experiment_id": experiment_id,
            "eval_id": eval_id,
            "mode": summary["mode"],
            "split": args.split,
            "metric_domain": args.metric_domain,
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "epoch": int(meta.epoch) if meta is not None else "",
            "eval_dir": str(output_dir),
            "seed": seed,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
            ),
            "threads": int(torch.get_num_threads()),
            "n_images": len(scored),
            "channels": channels,
            "eval_mode": eval_mode,
            "tile_size": tile_size,
            "params_total": params_summary.get("params_total", ""),
            "params_millions": params_summary.get("params_millions", ""),
            "psnr_mean": round(summary_metrics.get("psnr_mean", float("nan")), 4),
            "psnr_std": round(summary_metrics.get("psnr_std", float("nan")), 4),
            "ssim_mean": round(summary_metrics.get("ssim_mean", float("nan")), 5),
            "ssim_std": round(summary_metrics.get("ssim_std", float("nan")), 5),
            "lpips_mean": round(summary_metrics.get("lpips_mean", float("nan")), 5),
            "mae_mean": round(summary_metrics.get("mae_mean", float("nan")), 6),
            "rmse_mean": round(summary_metrics.get("rmse_mean", float("nan")), 6),
            "bicubic_psnr_mean": round(summary_metrics.get("bicubic_psnr_mean", float("nan")), 4),
            "bicubic_ssim_mean": round(summary_metrics.get("bicubic_ssim_mean", float("nan")), 5),
            "bicubic_lpips_mean": round(summary_metrics.get("bicubic_lpips_mean", float("nan")), 5),
            "bicubic_mae_mean": round(summary_metrics.get("bicubic_mae_mean", float("nan")), 6),
            "psnr_delta_mean": round(summary_metrics.get("psnr_delta_mean", float("nan")), 4),
            "psnr_delta_median": round(summary_metrics.get("psnr_delta_median", float("nan")), 4),
            "ssim_delta_mean": round(summary_metrics.get("ssim_delta_mean", float("nan")), 5),
            "win_rate_psnr": round(summary_metrics.get("win_rate_psnr", float("nan")), 4),
            "win_rate_ssim": round(summary_metrics.get("win_rate_ssim", float("nan")), 4),
            "worst_key": (failures["worst_psnr"][0]["key"] if failures["worst_psnr"] else ""),
            "worst_psnr": (
                round(float(failures["worst_psnr"][0]["psnr"]), 4) if failures["worst_psnr"] else ""
            ),
            "worst_delta_key": (failures["worst_delta"][0]["key"] if failures["worst_delta"] else ""),
            "worst_psnr_delta": (
                round(float(failures["worst_delta"][0].get("psnr_delta", float("nan"))), 4)
                if failures["worst_delta"]
                else ""
            ),
            "latency_mean_ms": round(latency.get("latency_mean_ms", float("nan")), 2),
            "latency_median_ms": round(latency.get("latency_median_ms", float("nan")), 2),
            "latency_p95_ms": round(latency.get("latency_p95_ms", float("nan")), 2),
            "latency_best_ms": round(latency.get("latency_best_ms", float("nan")), 2),
            "megapixels_per_second": round(latency.get("megapixels_per_second", float("nan")), 4),
            "bicubic_mean_ms": round(latency.get("bicubic_mean_ms", float("nan")), 2),
            "peak_memory": peak_memory,
            "eval_seconds": summary["eval_seconds"],
            "lpips_available": lpips_available,
            "torch_version": torch.__version__,
            "notes": f"{provenance} | maps={len(map_keys)} | failures={timing.get('failures', 0)}",
        }
        results_dir = Path(config["paths"]["results_csv"]).parent
        append_csv_row(results_dir / "evaluations.csv", eval_row, EVAL_CSV_COLUMNS)

        # The shared ledger keeps one table for every run, training or evaluation.
        ledger_row = {
            "timestamp": summary["timestamp"],
            "experiment_id": f"{experiment_id}__eval_{args.split}",
            "run_dir": str(output_dir),
            "status": f"eval:{summary['mode']}",
            "seed": seed,
            "device": str(device),
            "gpu_name": eval_row["gpu_name"],
            "epochs_completed": int(meta.epoch) if meta is not None else "",
            "params_total": params_summary.get("params_total", ""),
            "params_millions": params_summary.get("params_millions", ""),
            "channels": channels,
            "batch_size": 1,
            "train_mode": eval_mode,
            "degradation": describe_config(degradation),
            "best_epoch": int(meta.epoch) if meta is not None else "",
            "best_psnr": eval_row["psnr_mean"],
            "best_ssim": eval_row["ssim_mean"],
            "best_lpips": eval_row["lpips_mean"],
            "best_mae": eval_row["mae_mean"],
            "bicubic_psnr": eval_row["bicubic_psnr_mean"],
            "bicubic_ssim": eval_row["bicubic_ssim_mean"],
            "bicubic_lpips": eval_row["bicubic_lpips_mean"],
            "bicubic_mae": eval_row["bicubic_mae_mean"],
            "psnr_gain": eval_row["psnr_delta_mean"],
            "ssim_gain": eval_row["ssim_delta_mean"],
            "peak_memory": peak_memory,
            "torch_version": torch.__version__,
            "notes": (
                f"eval split={args.split} domain={args.metric_domain} n={len(scored)} "
                f"latency_mean_ms={eval_row['latency_mean_ms']} win_rate={eval_row['win_rate_psnr']}"
            ),
        }
        append_csv_row(config["paths"]["results_csv"], ledger_row, CSV_COLUMNS)

    LOGGER.info("=" * 78)
    LOGGER.info(
        "psnr    %.4f +/- %.4f   bicubic %.4f   delta %+.4f (median %+.4f)",
        summary_metrics.get("psnr_mean", float("nan")),
        summary_metrics.get("psnr_std", float("nan")),
        summary_metrics.get("bicubic_psnr_mean", float("nan")),
        summary_metrics.get("psnr_delta_mean", float("nan")),
        summary_metrics.get("psnr_delta_median", float("nan")),
    )
    LOGGER.info(
        "ssim    %.5f            bicubic %.5f   delta %+.5f",
        summary_metrics.get("ssim_mean", float("nan")),
        summary_metrics.get("bicubic_ssim_mean", float("nan")),
        summary_metrics.get("ssim_delta_mean", float("nan")),
    )
    LOGGER.info(
        "lpips   %.5f            bicubic %.5f",
        summary_metrics.get("lpips_mean", float("nan")),
        summary_metrics.get("bicubic_lpips_mean", float("nan")),
    )
    LOGGER.info(
        "mae     %.6f           bicubic %.6f",
        summary_metrics.get("mae_mean", float("nan")),
        summary_metrics.get("bicubic_mae_mean", float("nan")),
    )
    LOGGER.info(
        "win rate %.1f%% of %d image(s) beat bicubic on PSNR (%d regression(s))",
        100.0 * summary_metrics.get("win_rate_psnr", float("nan")),
        len(scored),
        failures.get("n_regressions", 0),
    )
    if latency:
        LOGGER.info(
            "latency  mean %.1f ms | median %.1f ms | p95 %.1f ms | %.2f MP/s | bicubic %.1f ms",
            latency.get("latency_mean_ms", float("nan")),
            latency.get("latency_median_ms", float("nan")),
            latency.get("latency_p95_ms", float("nan")),
            latency.get("megapixels_per_second", float("nan")),
            latency.get("bicubic_mean_ms", float("nan")),
        )
    if params_summary:
        LOGGER.info(
            "cost     %s params (%.4f M, %.2f MB fp32) | peak memory %s",
            params_summary["params_total"],
            params_summary["params_millions"],
            params_summary["fp32_size_mb"],
            peak_memory,
        )
    if failures["worst_psnr"]:
        worst = failures["worst_psnr"][0]
        LOGGER.info(
            "hardest  %s at %.3f dB (bicubic %.3f) -- downsampling destroys detail that "
            "no deterministic model can recover",
            worst["key"],
            float(worst["psnr"]),
            float(worst.get("bicubic_psnr", float("nan"))),
        )
    if not degradation.metadata.get("calibrated", False) and eval_mode == "synthetic":
        LOGGER.warning(
            "synthetic evaluation with an UNCALIBRATED degradation prior: these numbers "
            "describe our own forward model, not the official NoisyLR distribution"
        )
    LOGGER.info("summary  %s", summary_path)
    LOGGER.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
