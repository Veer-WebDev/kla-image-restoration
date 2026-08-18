"""Training loop.

Everything here is driven by ``configs/baseline.yaml`` plus ``--set key=value``
overrides, and every run appends one row to ``results/experiments.csv``. No number
reported anywhere in this project exists outside that ledger.

Design points that answer specific audit findings:

* ``torch.amp`` (not the deprecated ``torch.cuda.amp``), automatically disabled on
  CPU, so the same command runs on the evaluator's H100 and on a laptop (3.6).
* Deterministic seeding of the process, the sampler, and every DataLoader worker (3.7, 3.8).
* Epoch-aware training sampling, frozen validation (3.10).
* Self-describing checkpoints, resumable to the exact optimizer/scheduler state (3.13).
* Validation reports the bicubic baseline alongside the model on identical inputs,
  so 'did we actually beat interpolation' is answered every epoch, not at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import CheckpointMeta, load_checkpoint, save_checkpoint
from .dataset import (
    DatasetConfig,
    RestorationDataset,
    build_dataloader,
    detect_channels,
    discover_pairs,
    manifest,
    split_keys,
)
from .degradation import DegradationConfig, describe_config
from .extended_degradation import ExtendedDegradationConfig, describe_extended
from .metrics import LossConfig, aggregate, build_loss, compute_metrics, get_lpips
from .model import ModelConfig, build_model, model_summary
from .utils import (
    append_csv_row,
    deep_update,
    environment_snapshot,
    get_logger,
    human_bytes,
    load_yaml,
    seed_everything,
    select_device,
    setup_logging,
    to_numpy,
    write_json,
)

LOGGER = get_logger()

CSV_COLUMNS = [
    "timestamp",
    "experiment_id",
    "run_dir",
    "status",
    "seed",
    "device",
    "gpu_name",
    "epochs_requested",
    "epochs_completed",
    "params_total",
    "params_millions",
    "channels",
    "patch_size",
    "samples_per_image",
    "batch_size",
    "lr",
    "loss_kind",
    "ssim_weight",
    "train_mode",
    "synthetic_prob",
    "degradation",
    "best_epoch",
    "best_psnr",
    "best_ssim",
    "best_lpips",
    "best_mae",
    "bicubic_psnr",
    "bicubic_ssim",
    "bicubic_lpips",
    "bicubic_mae",
    "psnr_gain",
    "ssim_gain",
    "final_train_loss",
    "train_seconds",
    "seconds_per_epoch",
    "peak_memory",
    "torch_version",
    "notes",
]


@dataclass
class TrainState:
    """Mutable state carried across epochs, also what resume restores."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1
    best_metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def _coerce(value: str) -> Any:
    """Parse a CLI override value into a Python scalar/list."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [_coerce(v) for v in value[1:-1].split(",") if v.strip()]
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``--set a.b=value`` overrides onto a nested config dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node: dict[str, Any] = config
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
        LOGGER.info("override | %s = %r", key, node[parts[-1]])
    return config


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load the training config, merging a YAML file over the built-in defaults."""
    defaults: dict[str, Any] = {
        "experiment_id": "baseline_residual_unet",
        "seed": 42,
        "device": "auto",
        "strict_determinism": False,
        "data": {
            "gt_dir": "data/GT",
            "noisy_dir": None,
            "split_ratios": [0.8, 0.1, 0.1],
            "split_seed": 42,
            "channels": 1,
            "autodetect_channels": True,
            "patch_size": 256,
            "samples_per_image": 8,
            "synthetic_prob": 0.6,
            "augment": True,
            "augment_flips": True,
            "augment_rot90": True,
            "cache_images": True,
            "max_eval_size": 1024,
            "train_mode": "mixed",
            "eval_mode": "auto",
        },
        "model": ModelConfig().to_dict(),
        "loss": LossConfig().to_dict(),
        "train": {
            "epochs": 40,
            "batch_size": 8,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "betas": [0.9, 0.999],
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "min_lr_factor": 0.01,
            "grad_clip": 1.0,
            "amp": True,
            "num_workers": 2,
            "pin_memory": True,
            "eval_every": 1,
            "save_every": 1,
            "early_stop_patience": 0,
            "selection_metric": "psnr",
            "eval_lpips": True,
        },
        "inference": {
            "scale": 2,
            "batch_size": 4,
            "bit_depth": 8,
            "out_ext": ".png",
            "tile_size": 0,
            "tile_overlap": 32,
        },
        "paths": {"output_dir": "runs", "results_csv": "results/experiments.csv"},
        "degradation": {},
        "extended_degradation": {},
    }
    if path:
        file_config = load_yaml(path)
        defaults = deep_update(defaults, file_config)
    apply_overrides(defaults, overrides)
    return defaults


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW (default) with no weight decay on norms and biases."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": float(cfg["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kind = str(cfg.get("optimizer", "adamw")).lower()
    betas = tuple(float(b) for b in cfg.get("betas", (0.9, 0.999)))
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=float(cfg["lr"]), betas=betas)
    if kind == "adam":
        return torch.optim.Adam(groups, lr=float(cfg["lr"]), betas=betas)
    if kind == "sgd":
        return torch.optim.SGD(groups, lr=float(cfg["lr"]), momentum=0.9, nesterov=True)
    raise ValueError(f"unsupported optimizer {kind!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Cosine schedule with linear warmup, stepped per optimizer step."""
    kind = str(cfg.get("scheduler", "cosine")).lower()
    if kind in {"none", "constant"}:
        return None
    epochs = int(cfg["epochs"])
    total = max(1, epochs * max(1, steps_per_epoch))
    warmup = int(max(0, float(cfg.get("warmup_epochs", 0))) * max(1, steps_per_epoch))
    min_factor = float(cfg.get("min_lr_factor", 0.01))

    if kind == "cosine":

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total // 3), gamma=0.3
        )
    raise ValueError(f"unsupported scheduler {kind!r}")


def _batch_pairs(batch: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (noisy, gt) pairs with a batch dimension, stacked or not."""
    if batch["stacked"]:
        return [(batch["noisy"], batch["gt"])]
    return [(n[None], g[None]) for n, g in zip(batch["noisy"], batch["gt"])]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: dict[str, Any],
    state: TrainState,
) -> dict[str, float]:
    """One training epoch. Returns mean loss terms."""
    model.train()
    grad_clip = float(cfg.get("grad_clip", 0.0))
    amp_enabled = bool(scaler.is_enabled())
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    totals: dict[str, float] = {}
    n_batches = 0
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        for noisy, gt in _batch_pairs(batch):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(noisy, target_size=tuple(gt.shape[-2:]), clamp=False)
                loss, terms = loss_fn(pred.float(), gt)

            if not torch.isfinite(loss):
                LOGGER.warning("non-finite loss at step %d; batch skipped", state.global_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            state.global_step += 1

            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

        if batch_index % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            done = totals.get("total", 0.0) / max(1, n_batches)
            LOGGER.info(
                "epoch %d | batch %d/%d | loss %.5f | lr %.3e",
                state.epoch,
                batch_index,
                len(loader),
                done,
                lr,
            )
            print(
                f"JCODE_PROGRESS {json.dumps({'current': batch_index, 'total': len(loader), 'unit': 'batches', 'message': f'epoch {state.epoch} loss {done:.5f}'})}",
                flush=True,
            )

    out = {k: v / max(1, n_batches) for k, v in totals.items()}
    out["seconds"] = time.perf_counter() - started
    out["lr"] = optimizer.param_groups[0]["lr"]
    return out


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lpips_model: Any | None = None,
    include_bicubic: bool = True,
) -> dict[str, float]:
    """Evaluate on the frozen validation split, with the bicubic baseline alongside."""
    model.eval()
    model_rows: list[dict[str, float]] = []
    bicubic_rows: list[dict[str, float]] = []
    started = time.perf_counter()

    for batch in loader:
        for noisy, gt in _batch_pairs(batch):
            noisy = noisy.to(device, non_blocking=True)
            gt_dev = gt.to(device, non_blocking=True)
            target_size = tuple(gt_dev.shape[-2:])
            parts = model(noisy, target_size=target_size, clamp=True, return_parts=True)
            pred_np = to_numpy(parts["restored"][0].float().cpu())
            gt_np = to_numpy(gt[0].float())
            model_rows.append(
                compute_metrics(pred_np, gt_np, lpips_model=lpips_model, device=device)
            )
            if include_bicubic:
                base_np = to_numpy(parts["base"][0].float().clamp(0, 1).cpu())
                bicubic_rows.append(
                    compute_metrics(base_np, gt_np, lpips_model=lpips_model, device=device)
                )

    out = aggregate(model_rows)
    if include_bicubic and bicubic_rows:
        for key, value in aggregate(bicubic_rows).items():
            out[f"bicubic_{key}"] = value
    out["val_seconds"] = time.perf_counter() - started
    return out


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def split_keys_by_source_manifest(
    keys: Iterable[str],
    source_manifest: str | Path,
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split materialized sample keys by their clean-source SHA-256 group.

    A materialized corpus can contain many crop/degradation views of a single
    clean source. Splitting its filename keys directly would leak source content
    from train into validation. The materializer writes ``sample_id`` and
    ``source_sha256`` explicitly so this function can reject incomplete or
    ambiguous mappings before the first optimization step.
    """
    path = Path(source_manifest)
    if not path.is_file():
        raise FileNotFoundError(f"source manifest not found: {path}")
    available = set(keys)
    key_to_source: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_sha256"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"source manifest must contain {sorted(required)}: {path}")
        for row in reader:
            key = str(row["sample_id"]).strip().lower()
            source = str(row["source_sha256"]).strip().lower()
            if not key or not source:
                raise ValueError(f"blank sample_id or source_sha256 in {path}")
            if key in key_to_source and key_to_source[key] != source:
                raise ValueError(f"sample_id maps to multiple sources in {path}: {key}")
            key_to_source[key] = source
    missing = sorted(available - set(key_to_source))
    extra = sorted(set(key_to_source) - available)
    if missing or extra:
        raise ValueError(
            f"source manifest does not exactly match discovered pairs: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    source_to_keys: dict[str, list[str]] = {}
    for key in sorted(available):
        source_to_keys.setdefault(key_to_source[key], []).append(key)
    source_splits = split_keys(source_to_keys, ratios=ratios, seed=seed)
    sample_splits = {
        split: sorted(key for source in sources for key in source_to_keys[source])
        for split, sources in source_splits.items()
    }
    return sample_splits, source_splits


def prepare_data(
    config: dict[str, Any],
    degradation: DegradationConfig,
    extended: ExtendedDegradationConfig | None = None,
) -> tuple[RestorationDataset, RestorationDataset, dict[str, Any]]:
    """Discover, split and wrap the data. Fails loudly on an empty or unpaired dataset."""
    data_cfg = config["data"]
    extended = extended or ExtendedDegradationConfig()
    gt_dir = Path(data_cfg["gt_dir"])
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"GT directory not found: {gt_dir.resolve()}. "
            "Place the official ground-truth images there, or point data.gt_dir at them."
        )
    noisy_dir = data_cfg.get("noisy_dir")
    gt_map, noisy_map, report = discover_pairs(gt_dir, noisy_dir)
    if not gt_map:
        raise FileNotFoundError(f"no readable images in {gt_dir.resolve()}")
    report.log()

    channels = int(data_cfg["channels"])
    if data_cfg.get("autodetect_channels", True):
        detected = detect_channels(gt_map)
        if detected != channels:
            LOGGER.warning(
                "channel count from data is %d but config says %d; using %d",
                detected,
                channels,
                detected,
            )
            channels = detected
    config["model"]["in_channels"] = channels
    config["model"]["out_channels"] = channels
    data_cfg["channels"] = channels

    ratios = tuple(float(r) for r in data_cfg["split_ratios"])
    split_seed = int(data_cfg["split_seed"])
    source_manifest = data_cfg.get("source_manifest")
    if source_manifest:
        splits, source_splits = split_keys_by_source_manifest(
            gt_map.keys(), source_manifest, ratios=ratios, seed=split_seed
        )
        split_unit = "source_sha256_manifest"
    else:
        splits = split_keys(gt_map.keys(), ratios, split_seed)
        source_splits = splits
        split_unit = "canonical_stem"
    LOGGER.info(
        "split | train=%d val=%d test=%d (%s, seed=%d)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        split_unit,
        split_seed,
    )
    if not splits["train"] or not splits["val"]:
        raise ValueError(
            f"insufficient data: {len(gt_map)} source images produced "
            f"train={len(splits['train'])} val={len(splits['val'])}"
        )

    train_mode = str(data_cfg["train_mode"])
    if train_mode in {"official", "mixed"} and not noisy_map:
        LOGGER.warning("train_mode=%s requested but no NoisyLR files found; using synthetic", train_mode)
        train_mode = "synthetic"

    eval_mode = str(data_cfg.get("eval_mode", "auto"))
    if eval_mode == "auto":
        eval_mode = "official" if noisy_map else "synthetic"

    train_ds = RestorationDataset(
        splits["train"],
        gt_map,
        noisy_map,
        mode=train_mode,  # type: ignore[arg-type]
        degradation=degradation,
        extended=extended,
        config=DatasetConfig(
            patch_size=int(data_cfg["patch_size"]),
            samples_per_image=int(data_cfg["samples_per_image"]),
            synthetic_prob=float(data_cfg["synthetic_prob"]),
            augment=bool(data_cfg["augment"]),
            augment_flips=bool(data_cfg["augment_flips"]),
            augment_rot90=bool(data_cfg["augment_rot90"]),
            channels=channels,
            training=True,
            cache_images=bool(data_cfg["cache_images"]),
        ),
        seed=int(config["seed"]),
        on_missing="synthetic",
    )
    val_ds = RestorationDataset(
        splits["val"],
        gt_map,
        noisy_map,
        mode=eval_mode,  # type: ignore[arg-type]
        degradation=degradation,
        config=DatasetConfig(
            patch_size=int(data_cfg["patch_size"]),
            samples_per_image=1,
            synthetic_prob=0.0,
            augment=False,
            channels=channels,
            training=False,
            max_eval_size=data_cfg.get("max_eval_size"),
            cache_images=bool(data_cfg["cache_images"]),
        ),
        seed=int(config["seed"]) + 1,
        # Validation must never be silently padded with synthetic samples (audit 3.9).
        on_missing="error" if eval_mode == "official" else "synthetic",
    )
    split_info = {
        "pairing": report.summary(),
        "splits": {k: len(v) for k, v in splits.items()},
        "split_keys": splits,
        "source_splits": source_splits,
        "split_unit": split_unit,
        "source_manifest": str(source_manifest) if source_manifest else None,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "channels": channels,
    }
    return train_ds, val_ds, split_info


def train(config: dict[str, Any], resume: str | Path | None = None) -> dict[str, Any]:
    """Run training end to end and return the summary that goes into the CSV."""
    run_started = time.perf_counter()
    seed = int(config["seed"])
    seed_everything(seed, strict=bool(config.get("strict_determinism", False)))
    device = select_device(config.get("device", "auto"))

    experiment_id = str(config["experiment_id"])
    run_dir = Path(config["paths"]["output_dir"]) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    LOGGER.info("=" * 78)
    LOGGER.info("experiment %s | device=%s | seed=%d", experiment_id, device, seed)
    LOGGER.info("=" * 78)

    degradation = DegradationConfig.from_dict(config.get("degradation") or {})
    LOGGER.info("degradation | %s", describe_config(degradation))

    extended = ExtendedDegradationConfig.from_dict(config.get("extended_degradation") or {})
    LOGGER.info("degradation | %s", describe_extended(extended))

    train_ds, val_ds, split_info = prepare_data(config, degradation, extended)
    train_cfg = config["train"]
    num_workers = int(train_cfg["num_workers"])
    if device.type == "cpu" and num_workers > 0:
        # Worker processes on CPU-only machines usually cost more than they save here.
        LOGGER.info("cpu device | reducing num_workers %d -> 0", num_workers)
        num_workers = 0

    train_loader = build_dataloader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        pin_memory=bool(train_cfg["pin_memory"]) and device.type == "cuda",
        drop_last=len(train_ds) > int(train_cfg["batch_size"]),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        seed=seed + 1,
        pin_memory=False,
    )

    model = build_model(ModelConfig.from_dict(config["model"])).to(device)
    summary = model_summary(model)
    LOGGER.info(
        "model | %s | %.4fM params | %.2fMB fp32",
        model.config.name,
        summary["params_millions"],
        summary["fp32_size_mb"],
    )

    loss_fn = build_loss(LossConfig.from_dict(config["loss"])).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, len(train_loader))
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if bool(train_cfg["amp"]) and not amp_enabled:
        LOGGER.info("amp requested but device is %s; running in fp32", device.type)

    lpips_model = None
    if bool(train_cfg.get("eval_lpips", True)):
        try:
            lpips_model = get_lpips("alex", device)
            LOGGER.info("LPIPS enabled for validation (never used at inference)")
        except RuntimeError as exc:
            LOGGER.warning("LPIPS unavailable, continuing without it: %s", exc)

    state = TrainState()
    if resume:
        payload, meta = load_checkpoint(resume, map_location=device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if "scaler_state" in payload and amp_enabled:
            scaler.load_state_dict(payload["scaler_state"])
        state.epoch = int(meta.epoch)
        state.global_step = int(meta.global_step)
        state.history = list(payload.get("history", []))
        if meta.metrics:
            state.best_metrics = dict(meta.metrics)
            state.best_metric = float(meta.metrics.get(f"{train_cfg['selection_metric']}_mean", -math.inf))
        LOGGER.info("resumed from %s at epoch %d", resume, state.epoch)

    write_json(
        run_dir / "run_config.json",
        {
            "config": config,
            "split_info": {k: v for k, v in split_info.items() if k != "split_keys"},
            "model_summary": summary,
            "environment": environment_snapshot(),
        },
    )
    write_json(run_dir / "split.json", split_info["split_keys"])
    write_json(run_dir / "val_manifest.json", manifest(val_ds))

    selection = str(train_cfg["selection_metric"])
    higher_is_better = selection in {"psnr", "ssim"}
    patience = int(train_cfg.get("early_stop_patience", 0))
    epochs = int(train_cfg["epochs"])
    since_improved = 0
    final_train: dict[str, float] = {}
    bicubic_reference: dict[str, float] = {}

    for epoch in range(state.epoch + 1, epochs + 1):
        state.epoch = epoch
        train_ds.set_epoch(epoch)
        train_ds.reset_stats()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device, train_cfg, state
        )
        final_train = train_metrics
        LOGGER.info(
            "epoch %d/%d | train loss %.5f | %.1fs | sources %s",
            epoch,
            epochs,
            train_metrics.get("total", float("nan")),
            train_metrics["seconds"],
            train_ds.consumption_stats(),
        )

        record: dict[str, Any] = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        if epoch % int(train_cfg["eval_every"]) == 0 or epoch == epochs:
            val_metrics = validate(model, val_loader, device, lpips_model=lpips_model)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            if not bicubic_reference:
                bicubic_reference = {
                    k: v for k, v in val_metrics.items() if k.startswith("bicubic_")
                }
            psnr_gain = val_metrics.get("psnr_mean", float("nan")) - val_metrics.get(
                "bicubic_psnr_mean", float("nan")
            )
            LOGGER.info(
                "epoch %d | val psnr %.3f (bicubic %.3f, gain %+.3f) ssim %.4f lpips %s",
                epoch,
                val_metrics.get("psnr_mean", float("nan")),
                val_metrics.get("bicubic_psnr_mean", float("nan")),
                psnr_gain,
                val_metrics.get("ssim_mean", float("nan")),
                f"{val_metrics.get('lpips_mean', float('nan')):.4f}"
                if "lpips_mean" in val_metrics
                else "n/a",
            )

            key = f"{selection}_mean"
            current = float(val_metrics.get(key, float("nan")))
            if not math.isnan(current):
                # Normalize to "higher is better" so one comparison covers both cases.
                score = current if higher_is_better else -current
                if score > state.best_metric:
                    state.best_metric = score
                    state.best_epoch = epoch
                    state.best_metrics = dict(val_metrics)
                    since_improved = 0
                    save_checkpoint(
                        run_dir / "best.pth",
                        model,
                        CheckpointMeta(
                            experiment_id=experiment_id,
                            epoch=epoch,
                            global_step=state.global_step,
                            seed=seed,
                            channels=int(split_info["channels"]),
                            inference_scale=int(config["inference"]["scale"]),
                            bit_depth=int(config["inference"]["bit_depth"]),
                            model_config=model.config.to_dict(),
                            degradation_config=degradation.to_dict(),
                            dataset_config=train_ds.config.to_dict(),
                            loss_config=config["loss"],
                            train_config=train_cfg,
                            metrics={k: float(v) for k, v in val_metrics.items()},
                        ),
                    )
                    LOGGER.info("new best %s=%.5f at epoch %d -> best.pth", selection, current, epoch)
                else:
                    since_improved += 1

        state.history.append(record)
        append_csv_row(run_dir / "history.csv", record, list(record.keys()))

        if epoch % int(train_cfg["save_every"]) == 0:
            save_checkpoint(
                run_dir / "last.pth",
                model,
                CheckpointMeta(
                    experiment_id=experiment_id,
                    epoch=epoch,
                    global_step=state.global_step,
                    seed=seed,
                    channels=int(split_info["channels"]),
                    inference_scale=int(config["inference"]["scale"]),
                    bit_depth=int(config["inference"]["bit_depth"]),
                    model_config=model.config.to_dict(),
                    degradation_config=degradation.to_dict(),
                    dataset_config=train_ds.config.to_dict(),
                    loss_config=config["loss"],
                    train_config=train_cfg,
                    metrics=dict(state.best_metrics),
                ),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                history=state.history,
            )

        if patience and since_improved >= patience:
            LOGGER.info("early stop: no improvement in %d evaluations", since_improved)
            break

    elapsed = time.perf_counter() - run_started
    peak_memory = (
        human_bytes(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else "n/a"
    )
    best = state.best_metrics
    summary_row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "status": "completed",
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "epochs_requested": epochs,
        "epochs_completed": state.epoch,
        "params_total": summary["params_total"],
        "params_millions": summary["params_millions"],
        "channels": split_info["channels"],
        "patch_size": config["data"]["patch_size"],
        "samples_per_image": config["data"]["samples_per_image"],
        "batch_size": train_cfg["batch_size"],
        "lr": train_cfg["lr"],
        "loss_kind": config["loss"]["kind"],
        "ssim_weight": config["loss"]["ssim_weight"],
        "train_mode": split_info["train_mode"],
        "synthetic_prob": config["data"]["synthetic_prob"],
        "degradation": describe_config(degradation),
        "best_epoch": state.best_epoch,
        "best_psnr": round(best.get("psnr_mean", float("nan")), 4),
        "best_ssim": round(best.get("ssim_mean", float("nan")), 5),
        "best_lpips": round(best.get("lpips_mean", float("nan")), 5),
        "best_mae": round(best.get("mae_mean", float("nan")), 6),
        "bicubic_psnr": round(best.get("bicubic_psnr_mean", float("nan")), 4),
        "bicubic_ssim": round(best.get("bicubic_ssim_mean", float("nan")), 5),
        "bicubic_lpips": round(best.get("bicubic_lpips_mean", float("nan")), 5),
        "bicubic_mae": round(best.get("bicubic_mae_mean", float("nan")), 6),
        "psnr_gain": round(
            best.get("psnr_mean", float("nan")) - best.get("bicubic_psnr_mean", float("nan")), 4
        ),
        "ssim_gain": round(
            best.get("ssim_mean", float("nan")) - best.get("bicubic_ssim_mean", float("nan")), 5
        ),
        "final_train_loss": round(final_train.get("total", float("nan")), 6),
        "train_seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / max(1, state.epoch), 1),
        "peak_memory": peak_memory,
        "torch_version": torch.__version__,
        "notes": f"workers={num_workers} amp={amp_enabled}",
    }
    append_csv_row(config["paths"]["results_csv"], summary_row, CSV_COLUMNS)
    write_json(run_dir / "summary.json", summary_row)
    LOGGER.info("=" * 78)
    LOGGER.info(
        "done | best %s at epoch %s | psnr %.3f vs bicubic %.3f (%+.3f) | %.1fs",
        selection,
        state.best_epoch,
        summary_row["best_psnr"],
        summary_row["bicubic_psnr"],
        summary_row["psnr_gain"],
        elapsed,
    )
    LOGGER.info("=" * 78)
    return summary_row


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the KLA image restoration model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML training config")
    parser.add_argument("--degradation-config", default="configs/degradation.yaml", help="YAML degradation config")
    parser.add_argument("--gt-dir", default=None, help="override data.gt_dir")
    parser.add_argument("--noisy-dir", default=None, help="override data.noisy_dir")
    parser.add_argument("--experiment-id", default=None, help="override experiment_id")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="override train.lr")
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument("--device", default=None, help="auto | cuda | cpu")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--no-lpips", action="store_true", help="disable LPIPS during validation")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any nested config key, e.g. --set model.base_channels=48",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = load_config(args.config, args.set)
    degradation_path = Path(args.degradation_config)
    if degradation_path.exists():
        config["degradation"] = load_yaml(degradation_path)
    elif args.degradation_config != "configs/degradation.yaml":
        raise FileNotFoundError(f"degradation config not found: {degradation_path}")

    if args.gt_dir:
        config["data"]["gt_dir"] = args.gt_dir
    if args.noisy_dir:
        config["data"]["noisy_dir"] = args.noisy_dir
    if args.experiment_id:
        config["experiment_id"] = args.experiment_id
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.no_lpips:
        config["train"]["eval_lpips"] = False

    try:
        train(config, resume=args.resume)
    except KeyboardInterrupt:
        LOGGER.warning("interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.error("training failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
