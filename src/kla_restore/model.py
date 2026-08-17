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
    # Extra knobs used by the alternative architectures (ignored by residual_unet).
    # Kept on the shared config so every checkpoint stays self-describing and the
    # experiment CSV has one flat schema across model families.
    num_blocks: int = 8          # trunk depth for flat/NAFNet variants
    block_growth: int = 2        # channel multiplier per NAFNet stage

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
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.block_growth < 1:
            raise ValueError(f"block_growth must be >= 1, got {self.block_growth}")

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


class ResidualRestorer(nn.Module):
    """Shared NoisyLR -> bicubic upsample -> +residual -> clamp contract.

    Every restoration model in this project is a *residual learner on top of a
    fixed bicubic base*. That contract lives here so all architectures share one
    forward path, one size contract and one explainability interface
    (``return_parts``); subclasses only implement :meth:`residual`.

    Subclasses must:

    * build their layers in ``__init__`` after calling ``super().__init__(config)``;
    * set ``self._size_multiple`` (spatial multiple the trunk needs; 1 = anything);
    * implement ``residual(x, condition=None) -> Tensor`` returning a same-size
      correction for the already-upsampled input ``x``.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a ModelConfig or keyword arguments, not both")
        self.config = config
        self.condition_dim = config.condition_dim if config.degradation_aware else 0
        self._size_multiple = 1

    # ---------------------------------------------------------------- internals
    def _build_condition_mlp(self, width: int) -> None:
        """Attach a FiLM MLP that produces per-channel (gamma, beta) for ``width``."""
        if self.condition_dim:
            self.condition_mlp = nn.Sequential(
                nn.Linear(self.condition_dim, width),
                nn.GELU(),
                nn.Linear(width, 2 * width),
            )
        else:
            self.condition_mlp = None

    def _film(self, h: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_mlp is None:
            return h
        if condition is None:
            raise ValueError("model is degradation-aware but no condition was provided")
        gamma, beta = self.condition_mlp(condition.to(h.dtype)).chunk(2, dim=1)
        return h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        multiple = self._size_multiple
        h, w = x.shape[-2:]
        pad_h = (-h) % multiple
        pad_w = (-w) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > 1 else "replicate")
        return x, (pad_h, pad_w)

    def residual(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover - abstract

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


class ResidualUNet(ResidualRestorer):
    """U-Net that predicts a residual on top of an upsampled input.

    The original, audited baseline. Encoder/bottleneck/decoder with GroupNorm+GELU
    ConvBlocks and optional FiLM conditioning on the bottleneck.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        config = self.config
        self._size_multiple = 2**config.depth

        widths = [config.base_channels * (2**i) for i in range(config.depth)]
        bottleneck_width = config.base_channels * (2**config.depth)
        self._build_condition_mlp(bottleneck_width)

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

    def residual(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        padded, (pad_h, pad_w) = self._pad_to_multiple(x)

        skips: list[torch.Tensor] = []
        h = padded
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        h = self._film(h, condition)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if pad_h or pad_w:
            out = out[..., : out.shape[-2] - pad_h, : out.shape[-1] - pad_w]
        return out


# --------------------------------------------------------------------------- EDSR
class _ResBlock(nn.Module):
    """EDSR-style residual block: Conv-GELU-Conv with a residual scaling of 0.1."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + 0.1 * self.body(x)


class EDSRRestorer(ResidualRestorer):
    """Flat, full-resolution residual CNN (EDSR-style, no down/upsampling).

    A deliberately different inductive bias from the U-Net: no spatial pooling, so
    it never loses fine detail to strided operations, at the cost of processing
    everything at full resolution. ``num_blocks`` residual blocks at a constant
    width. Because there is no pooling it accepts any input size (multiple = 1).
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        config = self.config
        self._size_multiple = 1
        width = config.base_channels
        self._build_condition_mlp(width)
        self.head_in = nn.Conv2d(config.in_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(_ResBlock(width) for _ in range(config.num_blocks))
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.head = nn.Conv2d(width, config.out_channels, 3, padding=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.head_in(x)
        feat = self._film(feat, condition)
        h = feat
        for block in self.blocks:
            h = block(h)
        h = self.body_tail(h) + feat  # long skip, standard in EDSR trunks
        return self.head(h)


# ------------------------------------------------------------------------- NAFNet
class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (NAFNet normalization)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _NAFBlock(nn.Module):
    """NAFNet block: SimpleGate + simplified channel attention, no activations.

    SimpleGate (split channels, multiply halves) replaces GELU/ReLU; a squeeze
    channel-attention gate replaces self-attention. This is the core idea of
    "Simple Baselines for Image Restoration" (Chen et al., 2022), reimplemented
    from the paper description (not copied) so the repo stays license-clean.
    """

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        hidden = channels * expand
        self.norm1 = _LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        # SimpleGate halves the channels, so channel attention runs on hidden // 2.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden // 2, hidden // 2, 1),
        )
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)
        self.norm2 = _LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, hidden, 1)
        self.conv4 = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _simple_gate(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = self.conv1(self.norm1(x))
        y = self.conv_dw(y)
        y = self._simple_gate(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta
        y = self.conv3(self.norm2(x))
        y = self._simple_gate(y)
        y = self.conv4(y)
        return x + y * self.gamma


class NAFNetRestorer(ResidualRestorer):
    """NAFNet-style U-shaped restorer (SimpleGate, no nonlinear activations).

    A modern, strong restoration backbone with a different design philosophy from
    the audited U-Net: LayerNorm + SimpleGate + simplified channel attention, and
    learnable per-block residual scales. ``depth`` encoder/decoder stages with
    ``num_blocks`` NAF blocks at the bottleneck; channels grow by ``block_growth``.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        config = self.config
        self._size_multiple = 2**config.depth
        width = config.base_channels
        growth = config.block_growth

        self.intro = nn.Conv2d(config.in_channels, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        enc_widths: list[int] = []
        for _ in range(config.depth):
            self.encoders.append(_NAFBlock(ch))
            enc_widths.append(ch)
            self.downs.append(nn.Conv2d(ch, ch * growth, 2, stride=2))
            ch = ch * growth

        self.middle = nn.Sequential(*[_NAFBlock(ch) for _ in range(config.num_blocks)])
        self._build_condition_mlp(ch)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(enc_widths):
            self.ups.append(
                nn.Sequential(nn.Conv2d(ch, skip_ch * 4, 1), nn.PixelShuffle(2))
            )
            self.decoders.append(_NAFBlock(skip_ch))
            ch = skip_ch
        self.head = nn.Conv2d(width, config.out_channels, 3, padding=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        padded, (pad_h, pad_w) = self._pad_to_multiple(x)
        h = self.intro(padded)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            h = encoder(h)
            skips.append(h)
            h = down(h)
        h = self.middle(h)
        h = self._film(h, condition)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = decoder(h + skip)
        out = self.head(h)
        if pad_h or pad_w:
            out = out[..., : out.shape[-2] - pad_h, : out.shape[-1] - pad_w]
        return out


#: Model registry. ``build_model`` dispatches on ``ModelConfig.name`` so new
#: architectures are opt-in via config only, and checkpoints stay self-describing.
MODEL_REGISTRY: dict[str, type[ResidualRestorer]] = {
    "residual_unet": ResidualUNet,
    "edsr": EDSRRestorer,
    "nafnet": NAFNetRestorer,
}


def build_model(config: ModelConfig | dict[str, Any] | None = None) -> ResidualRestorer:
    """Construct the model from a config object or mapping."""
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    config = config or ModelConfig()
    try:
        cls = MODEL_REGISTRY[config.name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model name {config.name!r}; known: {known}") from exc
    return cls(config)


def model_summary(model: ResidualRestorer) -> dict[str, Any]:
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
