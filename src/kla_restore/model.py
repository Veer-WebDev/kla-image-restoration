"""Residual U-Net restoration model.

Formulation (unchanged from the audited baseline, audit 1.1 / section 6):

    restored = clamp(bicubic_upsample(NoisyLR) + unet(bicubic_upsample(NoisyLR)))

The network only predicts a correction to the bicubic estimate, so a failure mode
degrades toward the bicubic baseline rather than toward noise.

Corrections applied relative to the starter notebook:

* pad-to-multiple-of-2**depth at the model boundary, cropped back afterwards, so
  arbitrary and odd input sizes are safe (audit 3.4);
* clamping is a flag, disabled during training to keep gradients alive (audit 3.5);
* GroupNorm group count is derived from the channel count instead of hardcoded 8;
* the model owns the bicubic upsample step, giving a single place where the
  train-time and inference-time size contract is expressed (audit 3.12).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    """Largest divisor of ``channels`` that is <= ``preferred``."""
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1  # pragma: no cover - unreachable, 1 always divides


class ConvBlock(nn.Module):
    """Two Conv-GroupNorm-GELU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.block(x)


@dataclass
class ModelConfig:
    """Self-describing model configuration, embedded in every checkpoint."""

    name: str = "residual_unet"
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    upsample_mode: str = "bicubic"
    clamp_output: bool = False
    degradation_aware: bool = False
    condition_dim: int = 0

    def __post_init__(self) -> None:
        if self.base_channels < 4:
            raise ValueError(f"base_channels must be >= 4, got {self.base_channels}")
        if not 1 <= self.depth <= 6:
            raise ValueError(f"depth must lie in [1, 6], got {self.depth}")
        if self.upsample_mode not in {"bicubic", "bilinear", "nearest"}:
            raise ValueError(f"unsupported upsample_mode {self.upsample_mode!r}")
        if self.in_channels < 1 or self.out_channels < 1:
            raise ValueError("channel counts must be >= 1")
        if self.degradation_aware and self.condition_dim < 1:
            raise ValueError("degradation_aware requires condition_dim >= 1")

    @property
    def size_multiple(self) -> int:
        """Spatial multiple the network requires; inputs are padded up to it."""
        return 2**self.depth

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # noqa: SLF001
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**data)


class ResidualUNet(nn.Module):
    """U-Net that predicts a residual on top of an upsampled input.

    Parameters
    ----------
    config:
        A :class:`ModelConfig`. Passing the config (rather than loose kwargs)
        is what makes checkpoints self-describing.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a ModelConfig or keyword arguments, not both")
        self.config = config

        widths = [config.base_channels * (2**i) for i in range(config.depth)]
        bottleneck_width = config.base_channels * (2**config.depth)

        cond = config.condition_dim if config.degradation_aware else 0
        self.condition_dim = cond
        if cond:
            # FiLM-style conditioning on the bottleneck only: cheap, and it keeps the
            # unconditional path bit-identical when the branch is disabled.
            self.condition_mlp = nn.Sequential(
                nn.Linear(cond, bottleneck_width),
                nn.GELU(),
                nn.Linear(bottleneck_width, 2 * bottleneck_width),
            )
        else:
            self.condition_mlp = None

        self.encoders = nn.ModuleList()
        in_ch = config.in_channels
        for width in widths:
            self.encoders.append(ConvBlock(in_ch, width))
            in_ch = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-1], bottleneck_width)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = bottleneck_width
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(prev, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))
            prev = width
        self.head = nn.Conv2d(widths[0], config.out_channels, kernel_size=1)
        # Near-identity start: the residual branch begins almost silent, so early
        # training cannot be worse than bicubic. A small non-zero weight (rather than
        # exact zero) keeps gradients flowing into the trunk from the very first step.
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    # ---------------------------------------------------------------- internals
    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        multiple = self.config.size_multiple
        h, w = x.shape[-2:]
        pad_h = (-h) % multiple
        pad_w = (-w) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > 1 else "replicate")
        return x, (pad_h, pad_w)

    def residual(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the residual for an already-upsampled input."""
        padded, (pad_h, pad_w) = self._pad_to_multiple(x)

        skips: list[torch.Tensor] = []
        h = padded
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)

        if self.condition_mlp is not None:
            if condition is None:
                raise ValueError("model is degradation-aware but no condition was provided")
            gamma_beta = self.condition_mlp(condition.to(h.dtype))
            gamma, beta = gamma_beta.chunk(2, dim=1)
            h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if pad_h or pad_w:
            out = out[..., : out.shape[-2] - pad_h, : out.shape[-1] - pad_w]
        return out

    def upsample(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize ``x`` to ``size`` with the configured interpolation mode."""
        mode = self.config.upsample_mode
        kwargs: dict[str, Any] = {"size": tuple(int(s) for s in size), "mode": mode}
        if mode in {"bicubic", "bilinear"}:
            kwargs["align_corners"] = False
            kwargs["antialias"] = False
        return F.interpolate(x, **kwargs)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        noisy_lr: torch.Tensor,
        target_size: tuple[int, int] | None = None,
        *,
        scale: int | None = None,
        condition: torch.Tensor | None = None,
        clamp: bool | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Restore a batch.

        Exactly one of ``target_size`` or ``scale`` must determine the output size.
        ``target_size`` is used during training and local evaluation, where GT is
        available. ``scale`` is the inference-time contract (assumption A1): the
        output is ``input_size * scale``.

        Parameters
        ----------
        noisy_lr:
            ``(N, C, h, w)`` tensor, unclipped.
        target_size:
            Explicit output ``(H, W)``.
        scale:
            Integer upscaling factor, used when ``target_size`` is ``None``.
        condition:
            ``(N, condition_dim)`` conditioning vector for the degradation-aware
            variant.
        clamp:
            Override ``config.clamp_output``. Inference and evaluation pass ``True``.
        return_parts:
            Also return the bicubic base and the raw residual, for explainability.
        """
        if noisy_lr.ndim != 4:
            raise ValueError(f"expected a 4D tensor, got shape {tuple(noisy_lr.shape)}")
        if target_size is None:
            if scale is None:
                raise ValueError("provide target_size or scale")
            if int(scale) < 1:
                raise ValueError(f"scale must be >= 1, got {scale}")
            h, w = noisy_lr.shape[-2:]
            target_size = (int(h) * int(scale), int(w) * int(scale))

        base = self.upsample(noisy_lr, target_size)
        residual = self.residual(base, condition=condition)
        out = base + residual
        do_clamp = self.config.clamp_output if clamp is None else bool(clamp)
        restored = torch.clamp(out, 0.0, 1.0) if do_clamp else out

        if return_parts:
            return {
                "restored": restored,
                "base": base,
                "residual": residual,
                "unclamped": out,
            }
        return restored


def build_model(config: ModelConfig | dict[str, Any] | None = None) -> ResidualUNet:
    """Construct the model from a config object or mapping."""
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    return ResidualUNet(config or ModelConfig())


def model_summary(model: ResidualUNet) -> dict[str, Any]:
    """Return parameter counts and the config, for logs and the experiment CSV."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_millions": round(total / 1e6, 4),
        "fp32_size_mb": round(total * 4 / 1024**2, 3),
        "config": model.config.to_dict(),
    }
