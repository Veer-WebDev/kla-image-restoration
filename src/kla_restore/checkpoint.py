"""Self-describing checkpoints (audit finding 3.13).

A checkpoint carries everything needed to rebuild the exact inference pipeline:
model config, channel count, normalization policy, the inference scale contract,
the degradation config used for training, the seed, and the environment snapshot.
Loading therefore never requires the reader to know ``base_channels`` in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import CHECKPOINT_FORMAT_VERSION, __version__
from .model import ModelConfig, ResidualUNet, build_model
from .utils import environment_snapshot, get_logger

LOGGER = get_logger()


@dataclass
class CheckpointMeta:
    """Non-tensor payload stored alongside the weights."""

    format_version: int = CHECKPOINT_FORMAT_VERSION
    package_version: str = __version__
    experiment_id: str = "unknown"
    epoch: int = 0
    global_step: int = 0
    seed: int = 42
    channels: int = 1
    inference_scale: int = 2
    bit_depth: int = 8
    model_config: dict[str, Any] = field(default_factory=dict)
    degradation_config: dict[str, Any] = field(default_factory=dict)
    dataset_config: dict[str, Any] = field(default_factory=dict)
    loss_config: dict[str, Any] = field(default_factory=dict)
    train_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "package_version": self.package_version,
            "experiment_id": self.experiment_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "seed": self.seed,
            "channels": self.channels,
            "inference_scale": self.inference_scale,
            "bit_depth": self.bit_depth,
            "model_config": dict(self.model_config),
            "degradation_config": dict(self.degradation_config),
            "dataset_config": dict(self.dataset_config),
            "loss_config": dict(self.loss_config),
            "train_config": dict(self.train_config),
            "metrics": dict(self.metrics),
            "environment": dict(self.environment),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMeta":
        fields = set(cls.__dataclass_fields__)  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in fields})


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    meta: CheckpointMeta,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a portable checkpoint.

    Optimizer/scheduler/scaler state is included so training can resume exactly
    (audit: no resume capability). ``best.pth`` is written without them by the
    trainer to keep the deliverable small.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not meta.environment:
        meta.environment = environment_snapshot()
    if not meta.model_config and hasattr(model, "config"):
        meta.model_config = model.config.to_dict()  # type: ignore[union-attr]

    payload: dict[str, Any] = {
        "meta": meta.to_dict(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state"] = scaler.state_dict()
    if history is not None:
        payload["history"] = history

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    LOGGER.debug("checkpoint written | %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], CheckpointMeta]:
    """Load a checkpoint and validate its format version."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.0
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a kla_restore checkpoint")

    meta_dict = payload.get("meta")
    if meta_dict is None:
        # Tolerate the starter notebook's flat layout so old runs remain loadable.
        LOGGER.warning("checkpoint %s has no meta block; assuming notebook defaults", path)
        meta_dict = {
            "format_version": 1,
            "epoch": int(payload.get("epoch", 0)),
            "seed": int(payload.get("seed", 42)),
            "metrics": dict(payload.get("metrics", {})),
            "model_config": {"base_channels": 32, "depth": 4},
        }
    meta = CheckpointMeta.from_dict(meta_dict)
    if meta.format_version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {meta.format_version} is newer than supported "
            f"{CHECKPOINT_FORMAT_VERSION}; upgrade the package"
        )
    return payload, meta


def load_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    strict: bool = True,
) -> tuple[ResidualUNet, CheckpointMeta]:
    """Rebuild the model described by a checkpoint and load its weights."""
    payload, meta = load_checkpoint(path, map_location)
    config = ModelConfig.from_dict(meta.model_config or {})
    model = build_model(config)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=strict)
    if missing:
        LOGGER.warning("missing keys when loading weights: %s", list(missing)[:8])
    if unexpected:
        LOGGER.warning("unexpected keys when loading weights: %s", list(unexpected)[:8])
    model.eval()
    return model, meta
