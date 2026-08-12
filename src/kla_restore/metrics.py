"""Metrics and losses.

Reporting metrics (PSNR, SSIM, LPIPS, MAE) follow the KLA problem statement.
LPIPS is **evaluation-only and lazily imported** so that ``inference.py`` never
depends on it or on network access (audit 3.14, assumption A15).

No classification metrics appear anywhere in this project: the task is restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_logger

LOGGER = get_logger()

_LPIPS_CACHE: dict[tuple[str, str], Any] = {}


# --------------------------------------------------------------------------------------
# pixel metrics (numpy, on saved-quality arrays)
# --------------------------------------------------------------------------------------
def _prepare(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 2:
        p = p[:, :, None]
    if t.ndim == 2:
        t = t[:, :, None]
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: prediction {p.shape} vs target {t.shape}")
    return p, t


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Returns ``inf`` for identical images."""
    p, t = _prepare(pred, target)
    mse = float(np.mean((p - t) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error."""
    p, t = _prepare(pred, target)
    return float(np.mean(np.abs(p - t)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    p, t = _prepare(pred, target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity via scikit-image, with correct multichannel handling."""
    from skimage.metrics import structural_similarity

    p, t = _prepare(pred, target)
    win = min(7, p.shape[0], p.shape[1])
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    if p.shape[2] > 1:
        value = structural_similarity(t, p, data_range=data_range, channel_axis=2, win_size=win)
    else:
        value = structural_similarity(t[:, :, 0], p[:, :, 0], data_range=data_range, win_size=win)
    return float(value)


# --------------------------------------------------------------------------------------
# LPIPS (evaluation only)
# --------------------------------------------------------------------------------------
def get_lpips(net: str = "alex", device: str | torch.device = "cpu") -> Any:
    """Return a cached LPIPS model.

    Raises
    ------
    RuntimeError
        If the ``lpips`` package or its pretrained weights are unavailable. Callers
        must treat LPIPS as optional and continue reporting the other metrics.
    """
    key = (str(net), str(device))
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips as lpips_pkg
    except ImportError as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "lpips is not installed; install it for evaluation or pass --no-lpips"
        ) from exc
    try:
        model = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
    except Exception as exc:  # pragma: no cover - needs network on first use
        raise RuntimeError(
            f"failed to initialize LPIPS(net={net!r}); pretrained weights may be unavailable "
            "offline. Evaluation can proceed with --no-lpips."
        ) from exc
    for param in model.parameters():
        param.requires_grad_(False)
    _LPIPS_CACHE[key] = model
    return model


def lpips_distance(
    pred: np.ndarray,
    target: np.ndarray,
    model: Any,
    device: str | torch.device = "cpu",
) -> float:
    """LPIPS between two [0, 1] arrays. Grayscale inputs are replicated to 3 channels."""
    p, t = _prepare(pred, target)
    p = np.clip(p, 0.0, 1.0).astype(np.float32)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    if p.shape[2] == 1:
        p = np.repeat(p, 3, axis=2)
        t = np.repeat(t, 3, axis=2)
    pt = torch.from_numpy(p.transpose(2, 0, 1))[None].to(device)
    tt = torch.from_numpy(t.transpose(2, 0, 1))[None].to(device)
    with torch.inference_mode():
        value = model(pt * 2.0 - 1.0, tt * 2.0 - 1.0)
    return float(value.reshape(-1)[0].item())


# --------------------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------------------
def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    lpips_model: Any | None = None,
    device: str | torch.device = "cpu",
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute the full reporting metric set for one image pair."""
    out = {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
    }
    if lpips_model is not None:
        try:
            out["lpips"] = lpips_distance(pred, target, lpips_model, device)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("LPIPS failed for one image: %s", exc)
            out["lpips"] = float("nan")
    return out


def aggregate(rows: Sequence[dict[str, float]], keys: Iterable[str] | None = None) -> dict[str, float]:
    """Mean and standard deviation per metric, ignoring NaN and infinite values."""
    if not rows:
        return {}
    metric_names = list(keys) if keys is not None else sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for name in metric_names:
        values = np.array(
            [row[name] for row in rows if name in row and np.isfinite(row[name])],
            dtype=np.float64,
        )
        if values.size == 0:
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_std"] = float("nan")
            continue
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out[f"{name}_min"] = float(values.min())
        out[f"{name}_max"] = float(values.max())
    out["n"] = float(len(rows))
    return out


# --------------------------------------------------------------------------------------
# torch losses
# --------------------------------------------------------------------------------------
def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Charbonnier (smooth L1-like) loss: ``sqrt((pred - target)^2 + eps^2)``."""
    return torch.sqrt((pred - target) ** 2 + epsilon**2).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable mean SSIM over a batch.

    Uses a Gaussian window, matching the standard Wang et al. formulation. Window
    size adapts down for small patches so the function never fails on tiny inputs.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    channels = pred.shape[1]
    win = min(window_size, pred.shape[-1], pred.shape[-2])
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    window = _gaussian_window(win, sigma, channels, pred.device, pred.dtype)
    pad = win // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


@dataclass
class LossConfig:
    """Loss composition. Every weight is an ablation knob."""

    kind: str = "charbonnier_ssim"
    charbonnier_epsilon: float = 1e-3
    ssim_weight: float = 0.2
    l1_weight: float = 0.0

    def __post_init__(self) -> None:
        allowed = {"l1", "l2", "charbonnier", "charbonnier_ssim"}
        if self.kind not in allowed:
            raise ValueError(f"loss kind must be one of {sorted(allowed)}, got {self.kind!r}")
        if self.ssim_weight < 0 or self.l1_weight < 0:
            raise ValueError("loss weights must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "charbonnier_epsilon": self.charbonnier_epsilon,
            "ssim_weight": self.ssim_weight,
            "l1_weight": self.l1_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LossConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)  # noqa: SLF001
        if unknown:
            raise ValueError(f"unknown loss config keys: {sorted(unknown)}")
        return cls(**data)


class RestorationLoss(torch.nn.Module):
    """Composite restoration loss with per-term reporting."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        parts: dict[str, float] = {}
        if cfg.kind == "l1":
            total = F.l1_loss(pred, target)
            parts["l1"] = float(total.detach())
        elif cfg.kind == "l2":
            total = F.mse_loss(pred, target)
            parts["l2"] = float(total.detach())
        else:
            total = charbonnier_loss(pred, target, cfg.charbonnier_epsilon)
            parts["charbonnier"] = float(total.detach())
            if cfg.kind == "charbonnier_ssim" and cfg.ssim_weight > 0:
                # SSIM is computed on clamped values: it is only meaningful in [0, 1].
                ssim_value = ssim_torch(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
                parts["ssim"] = float(ssim_value.detach())
                total = total + cfg.ssim_weight * (1.0 - ssim_value)
        if cfg.l1_weight > 0:
            l1 = F.l1_loss(pred, target)
            parts["l1"] = float(l1.detach())
            total = total + cfg.l1_weight * l1
        parts["total"] = float(total.detach())
        return total, parts


def build_loss(config: LossConfig | dict[str, Any] | None = None) -> RestorationLoss:
    """Construct the loss from a config object or mapping."""
    if isinstance(config, dict):
        config = LossConfig.from_dict(config)
    return RestorationLoss(config)
