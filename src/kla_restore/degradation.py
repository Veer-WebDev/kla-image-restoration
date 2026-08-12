"""Forward degradation model: additive Gaussian noise, multiplicative speckle, downsampling.

The KLA problem statement defines exactly three mechanisms and states that their
application order is not disclosed. This module therefore:

* implements only those three mechanisms -- no blur, no JPEG, no motion (audit 1.2);
* supports all six orderings explicitly, either sampled or pinned (audit 1.3);
* derives every random draw from a seed, so ``(source_id, sample_index, seed)``
  reproduces a sample bit-for-bit (audit 1.4, A10);
* draws an independent noise realization per sample -- no noise array is ever reused;
* never clips the NoisyLR result, matching "values may extend slightly outside [0,1]".

All ranges live in ``configs/degradation.yaml``; the defaults below mirror that file
so the module is usable standalone and testable without file IO.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import derive_seed, get_logger

#: The three official degradation operations.
OPERATIONS: tuple[str, str, str] = ("gaussian", "speckle", "downsample")

#: All six orderings of the three operations, in a fixed canonical order so that
#: ``ORDER_PERMUTATIONS[i]`` is stable across runs and machines.
ORDER_PERMUTATIONS: tuple[tuple[str, str, str], ...] = tuple(itertools.permutations(OPERATIONS))

#: Downsampling kernels supported by :func:`downsample`.
KERNELS: tuple[str, ...] = ("area", "bicubic", "bilinear", "nearest", "lanczos")

_PIL_FILTERS = {
    "area": "BOX",
    "bicubic": "BICUBIC",
    "bilinear": "BILINEAR",
    "nearest": "NEAREST",
    "lanczos": "LANCZOS",
}


@dataclass(frozen=True)
class DegradationConfig:
    """Parameters of the forward degradation model.

    Attributes
    ----------
    gaussian_sigma:
        Inclusive range for the additive noise standard deviation, in normalized
        [0, 1] units (assumption A6/A9).
    speckle_sigma:
        Inclusive range for the multiplicative speckle standard deviation
        (assumption A5: ``x * (1 + N(0, sigma))``).
    scales:
        Candidate integer downsampling factors.
    scale_weights:
        Optional sampling weights for ``scales``; uniform when ``None``.
    kernels:
        Candidate downsampling kernels (assumption A7).
    kernel_weights:
        Optional sampling weights for ``kernels``; uniform when ``None``.
    orders:
        Which of the six permutations may be sampled. ``None`` means all six.
    fixed_order:
        Pin the order (used by the degradation-order ablation).
    gaussian_prob, speckle_prob:
        Probability that the corresponding noise operation is applied at all.
        Downsampling is always applied because the task is defined on NoisyLR.
    clip_noisy:
        Must stay ``False`` for KLA-faithful data.
    """

    gaussian_sigma: tuple[float, float] = (0.005, 0.08)
    speckle_sigma: tuple[float, float] = (0.01, 0.15)
    scales: tuple[int, ...] = (2, 4)
    scale_weights: tuple[float, ...] | None = None
    kernels: tuple[str, ...] = ("area", "bicubic", "bilinear")
    kernel_weights: tuple[float, ...] | None = None
    orders: tuple[tuple[str, str, str], ...] | None = None
    fixed_order: tuple[str, str, str] | None = None
    gaussian_prob: float = 1.0
    speckle_prob: float = 1.0
    clip_noisy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("gaussian_sigma", "speckle_sigma"):
            lo, hi = getattr(self, name)
            if not (0.0 <= float(lo) <= float(hi)):
                raise ValueError(f"{name} must satisfy 0 <= lo <= hi, got {(lo, hi)}")
        if not self.scales:
            raise ValueError("scales must not be empty")
        for scale in self.scales:
            if int(scale) < 1:
                raise ValueError(f"scale must be >= 1, got {scale}")
        if not self.kernels:
            raise ValueError("kernels must not be empty")
        for kernel in self.kernels:
            if kernel not in KERNELS:
                raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
        for weights, values, name in (
            (self.scale_weights, self.scales, "scale_weights"),
            (self.kernel_weights, self.kernels, "kernel_weights"),
        ):
            if weights is not None:
                if len(weights) != len(values):
                    raise ValueError(f"{name} length {len(weights)} != {len(values)}")
                if any(w < 0 for w in weights) or sum(weights) <= 0:
                    raise ValueError(f"{name} must be non-negative with a positive sum")
        if self.fixed_order is not None and tuple(self.fixed_order) not in ORDER_PERMUTATIONS:
            raise ValueError(f"fixed_order {self.fixed_order} is not a permutation of {OPERATIONS}")
        if self.orders is not None:
            if not self.orders:
                raise ValueError("orders must not be empty when provided")
            for order in self.orders:
                if tuple(order) not in ORDER_PERMUTATIONS:
                    raise ValueError(f"invalid order {order}")
        for prob_name in ("gaussian_prob", "speckle_prob"):
            prob = float(getattr(self, prob_name))
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"{prob_name} must lie in [0, 1], got {prob}")

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DegradationConfig":
        """Build a config from a plain mapping (e.g. parsed YAML)."""
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass API
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown degradation config keys: {sorted(unknown)}")

        def _range(key: str, default: tuple[float, float]) -> tuple[float, float]:
            value = data.get(key, default)
            if isinstance(value, (int, float)):
                return (float(value), float(value))
            seq = tuple(float(v) for v in value)
            if len(seq) != 2:
                raise ValueError(f"{key} must be a scalar or a 2-element range, got {value!r}")
            return seq

        orders = data.get("orders")
        parsed_orders: tuple[tuple[str, str, str], ...] | None = None
        if orders:
            parsed_orders = tuple(tuple(str(op) for op in order) for order in orders)  # type: ignore[misc]
        fixed = data.get("fixed_order")
        parsed_fixed = tuple(str(op) for op in fixed) if fixed else None

        return cls(
            gaussian_sigma=_range("gaussian_sigma", (0.005, 0.08)),
            speckle_sigma=_range("speckle_sigma", (0.01, 0.15)),
            scales=tuple(int(s) for s in data.get("scales", (2, 4))),
            scale_weights=(
                tuple(float(w) for w in data["scale_weights"]) if data.get("scale_weights") else None
            ),
            kernels=tuple(str(k) for k in data.get("kernels", ("area", "bicubic", "bilinear"))),
            kernel_weights=(
                tuple(float(w) for w in data["kernel_weights"]) if data.get("kernel_weights") else None
            ),
            orders=parsed_orders,
            fixed_order=parsed_fixed,  # type: ignore[arg-type]
            gaussian_prob=float(data.get("gaussian_prob", 1.0)),
            speckle_prob=float(data.get("speckle_prob", 1.0)),
            clip_noisy=bool(data.get("clip_noisy", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoints and run snapshots."""
        return {
            "gaussian_sigma": list(self.gaussian_sigma),
            "speckle_sigma": list(self.speckle_sigma),
            "scales": list(self.scales),
            "scale_weights": list(self.scale_weights) if self.scale_weights else None,
            "kernels": list(self.kernels),
            "kernel_weights": list(self.kernel_weights) if self.kernel_weights else None,
            "orders": [list(o) for o in self.orders] if self.orders else None,
            "fixed_order": list(self.fixed_order) if self.fixed_order else None,
            "gaussian_prob": self.gaussian_prob,
            "speckle_prob": self.speckle_prob,
            "clip_noisy": self.clip_noisy,
            "metadata": dict(self.metadata),
        }

    def with_overrides(self, **kwargs: Any) -> "DegradationConfig":
        """Return a copy with fields replaced (used by ablations)."""
        return replace(self, **kwargs)

    @property
    def allowed_orders(self) -> tuple[tuple[str, str, str], ...]:
        """Orders this config may sample from."""
        if self.fixed_order is not None:
            return (tuple(self.fixed_order),)  # type: ignore[return-value]
        return self.orders or ORDER_PERMUTATIONS


# --------------------------------------------------------------------------------------
# primitive operations
# --------------------------------------------------------------------------------------
def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean i.i.d. Gaussian noise. No clipping."""
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=image.shape)
    return (image + noise).astype(np.float32)


def add_speckle_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative speckle ``x * (1 + N(0, sigma))``. No clipping.

    This is the canonical ``imnoise(..., 'speckle')`` form (assumption A5).
    """
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=image.shape)
    return (image * (1.0 + noise)).astype(np.float32)


def downsample(image: np.ndarray, scale: int, kernel: str = "area") -> np.ndarray:
    """Downsample ``(H, W, C)`` by an integer factor using a PIL resampling filter.

    ``area`` maps to PIL ``BOX`` (true pixel averaging). Values are preserved in
    float32; PIL resizes each channel as an ``F`` mode image so no quantization
    occurs and out-of-range values survive intact.
    """
    from PIL import Image as PILImage

    scale = int(scale)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return image.astype(np.float32, copy=True)
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    height, width = arr.shape[:2]
    new_h = max(1, height // scale)
    new_w = max(1, width // scale)
    resample = getattr(PILImage.Resampling, _PIL_FILTERS[kernel])

    planes = []
    for channel in range(arr.shape[2]):
        plane = PILImage.fromarray(arr[:, :, channel], mode="F")
        planes.append(np.asarray(plane.resize((new_w, new_h), resample=resample), dtype=np.float32))
    return np.ascontiguousarray(np.stack(planes, axis=2))


# --------------------------------------------------------------------------------------
# sampling and application
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DegradationParams:
    """The concrete, resolved parameters used for one sample."""

    order: tuple[str, str, str]
    gaussian_sigma: float
    speckle_sigma: float
    scale: int
    kernel: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": "->".join(self.order),
            "gaussian_sigma": round(float(self.gaussian_sigma), 6),
            "speckle_sigma": round(float(self.speckle_sigma), 6),
            "scale": int(self.scale),
            "kernel": self.kernel,
            "seed": int(self.seed),
        }


def _choice(rng: np.random.Generator, values: Sequence[Any], weights: Sequence[float] | None) -> Any:
    if weights is None:
        return values[int(rng.integers(0, len(values)))]
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()
    return values[int(rng.choice(len(values), p=probs))]


def sample_params(config: DegradationConfig, seed: int) -> DegradationParams:
    """Draw a reproducible parameter set for one sample.

    The same ``(config, seed)`` always yields the same parameters, and the returned
    ``seed`` is what :func:`apply_degradations` uses for its noise realizations, so
    a sample is fully described by its params.
    """
    rng = np.random.default_rng(int(seed) % (2**63))
    orders = config.allowed_orders
    order = orders[int(rng.integers(0, len(orders)))]

    g_lo, g_hi = config.gaussian_sigma
    s_lo, s_hi = config.speckle_sigma
    gaussian_sigma = float(rng.uniform(g_lo, g_hi)) if rng.random() < config.gaussian_prob else 0.0
    speckle_sigma = float(rng.uniform(s_lo, s_hi)) if rng.random() < config.speckle_prob else 0.0
    scale = int(_choice(rng, config.scales, config.scale_weights))
    kernel = str(_choice(rng, config.kernels, config.kernel_weights))

    return DegradationParams(
        order=tuple(order),  # type: ignore[arg-type]
        gaussian_sigma=gaussian_sigma,
        speckle_sigma=speckle_sigma,
        scale=scale,
        kernel=kernel,
        seed=int(seed),
    )


def apply_degradations(
    image: np.ndarray,
    params: DegradationParams,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Apply the three degradations in ``params.order``.

    Parameters
    ----------
    image:
        Clean GT image, ``(H, W, C)`` float32 in [0, 1].
    params:
        Resolved parameters from :func:`sample_params`.
    clip:
        Leave ``False`` for KLA-faithful NoisyLR.

    Returns
    -------
    numpy.ndarray
        The degraded image. Its spatial size is ``(H // scale, W // scale)``
        regardless of where downsampling occurs in the order.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # A dedicated generator per sample: independent realizations, never reused.
    rng = np.random.default_rng((int(params.seed) ^ 0x9E3779B9) % (2**63))

    for op in params.order:
        if op == "gaussian":
            arr = add_gaussian_noise(arr, params.gaussian_sigma, rng)
        elif op == "speckle":
            arr = add_speckle_noise(arr, params.speckle_sigma, rng)
        elif op == "downsample":
            arr = downsample(arr, params.scale, params.kernel)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown operation {op!r}")

    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray(arr.astype(np.float32))


def degrade(
    image: np.ndarray,
    config: DegradationConfig,
    seed: int,
) -> tuple[np.ndarray, DegradationParams]:
    """Convenience wrapper: sample params from ``seed`` then apply them."""
    params = sample_params(config, seed)
    noisy = apply_degradations(image, params, clip=config.clip_noisy)
    return noisy, params


def sample_seed(master_seed: int, source_id: str, sample_index: int, epoch: int = 0) -> int:
    """Derive a per-sample seed.

    Including ``epoch`` makes training augmentation a genuine distribution rather
    than a fixed dataset (audit finding 3.10), while callers that pass ``epoch=0``
    -- validation and test -- get a frozen, reproducible set.
    """
    return derive_seed(int(master_seed), str(source_id), int(sample_index), int(epoch))


def order_matrix(
    image: np.ndarray,
    config: DegradationConfig,
    seed: int,
    orders: Iterable[Sequence[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Degrade one image once per ordering, holding all other parameters fixed.

    Used by the degradation-order ablation so the only varying factor is the order.
    """
    base = sample_params(config, seed)
    chosen = tuple(tuple(o) for o in (orders or ORDER_PERMUTATIONS))
    out: dict[str, np.ndarray] = {}
    for order in chosen:
        params = DegradationParams(
            order=order,  # type: ignore[arg-type]
            gaussian_sigma=base.gaussian_sigma,
            speckle_sigma=base.speckle_sigma,
            scale=base.scale,
            kernel=base.kernel,
            seed=base.seed,
        )
        out["->".join(order)] = apply_degradations(image, params, clip=config.clip_noisy)
    return out


def describe_config(config: DegradationConfig) -> str:
    """Human-readable one-line summary for logs."""
    return (
        f"gaussian={config.gaussian_sigma} speckle={config.speckle_sigma} "
        f"scales={config.scales} kernels={config.kernels} "
        f"orders={len(config.allowed_orders)}/6 clip={config.clip_noisy}"
    )


def log_config(config: DegradationConfig) -> None:
    """Log the degradation configuration at INFO level."""
    get_logger().info("degradation | %s", describe_config(config))
