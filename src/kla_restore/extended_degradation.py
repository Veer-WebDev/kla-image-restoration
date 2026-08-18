"""Extended SEM acquisition-artifact model (opt-in robustness augmentation).

The KLA restoration brief defines exactly three degradation mechanisms -- additive
Gaussian noise, multiplicative speckle and downsampling -- and :mod:`kla_restore.degradation`
implements *only* those, faithfully. This module is deliberately separate: it adds the
broader family of real SEM acquisition artifacts described in the SEMICON India
"Drift-Sense" dataset methodology so a model can be trained to be *tolerant* to them.

Nothing here changes the KLA-faithful path. The extended artifacts are applied only
when a training config explicitly enables them, and they act on the already-degraded
NoisyLR as an additional, seeded, reproducible augmentation. Defaults leave every
mechanism off, so importing this module never perturbs the strict submission pipeline.

Mechanisms (each independently toggled, each drawn from a seeded generator):

* **beam_blur** -- Gaussian point-spread function of the electron beam spot, with an
  ``astigmatism_ratio`` that stretches the PSF so one axis is sharper than the other.
* **shot_noise** -- Poisson electron-count noise. ``dose`` stands in for electron count:
  higher dose means lower relative noise (noise ~ 1/sqrt(dose)).
* **detector_noise** -- additive Gaussian readout noise, independent of dose.
* **vignette** -- radial darkening ``I * (1 - strength * r**2)``.
* **gamma** -- nonlinear detector gain ``I ** gamma``.
* **barrel** -- barrel/pincushion lens distortion, sampling the source at a displaced
  radius ``r' = r * (1 + k * r**2)``.
* **charging** -- sparse bright horizontal streaks from local sample charging.
* **drift_jitter** -- progressive row-to-row shear (drift) plus per-row horizontal
  jitter (vibration), a characteristic raster-scan artifact.

All operations preserve the ``(H, W, C)`` float32 convention and never clip, matching
the KLA rule that NoisyLR values may extend slightly outside ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: Canonical application order (an imaging-plausible chain). Each stage is skipped
#: unless its probability draw fires and its config block is present.
EXTENDED_STAGES: tuple[str, ...] = (
    "beam_blur",
    "shot_noise",
    "detector_noise",
    "vignette",
    "gamma",
    "barrel",
    "charging",
    "drift_jitter",
)


def _as_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    seq = tuple(float(v) for v in value)
    if len(seq) != 2:
        raise ValueError(f"expected a scalar or 2-element range, got {value!r}")
    if seq[0] > seq[1]:
        raise ValueError(f"range lo>hi: {seq}")
    return seq


@dataclass(frozen=True)
class ExtendedDegradationConfig:
    """Toggle + parameter ranges for each extended SEM artifact.

    Every ``*_prob`` is the probability that the stage is applied to a given sample.
    With all probabilities at their default of ``0.0`` this config is a no-op, so it
    is safe to attach to any pipeline without changing behaviour.
    """

    enabled: bool = False

    # Beam-spot PSF blur (Gaussian). Sigma in LR pixels.
    beam_blur_prob: float = 0.0
    beam_sigma: tuple[float, float] = (0.4, 1.2)
    astigmatism_ratio: tuple[float, float] = (1.0, 2.4)

    # Poisson shot noise. `dose` is a proxy for electron count.
    shot_noise_prob: float = 0.0
    dose: tuple[float, float] = (150.0, 4000.0)

    # Detector/readout additive Gaussian noise, in [0,1] units.
    detector_noise_prob: float = 0.0
    detector_sigma: tuple[float, float] = (0.005, 0.03)

    # Radial vignetting.
    vignette_prob: float = 0.0
    vignette_strength: tuple[float, float] = (0.1, 0.4)

    # Detector gamma (contrast miscalibration).
    gamma_prob: float = 0.0
    gamma: tuple[float, float] = (0.7, 1.8)

    # Barrel/pincushion distortion. Positive k = barrel, negative = pincushion.
    barrel_prob: float = 0.0
    barrel_k: tuple[float, float] = (-0.12, 0.12)

    # Charging streaks: sparse bright horizontal bands.
    charging_prob: float = 0.0
    charging_rate: tuple[float, float] = (0.02, 0.10)  # streaks per 100 rows
    charging_gain: tuple[float, float] = (0.15, 0.5)

    # Raster drift (progressive shear) + per-row jitter, in pixels.
    drift_jitter_prob: float = 0.0
    shear_px: tuple[float, float] = (0.0, 3.0)
    jitter_px: tuple[float, float] = (0.0, 1.0)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "beam_blur_prob",
            "shot_noise_prob",
            "detector_noise_prob",
            "vignette_prob",
            "gamma_prob",
            "barrel_prob",
            "charging_prob",
            "drift_jitter_prob",
        ):
            p = float(getattr(self, name))
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {p}")

    @property
    def any_active(self) -> bool:
        """True if enabled and at least one stage has a positive probability."""
        if not self.enabled:
            return False
        return any(
            float(getattr(self, f"{stage}_prob")) > 0.0 for stage in EXTENDED_STAGES
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExtendedDegradationConfig":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # noqa: SLF001
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown extended degradation keys: {sorted(unknown)}")

        def rng(key: str) -> tuple[float, float]:
            default = getattr(cls, "__dataclass_fields__")[key].default
            return _as_range(data.get(key), default)

        def prob(key: str) -> float:
            return float(data.get(key, 0.0))

        return cls(
            enabled=bool(data.get("enabled", False)),
            beam_blur_prob=prob("beam_blur_prob"),
            beam_sigma=rng("beam_sigma"),
            astigmatism_ratio=rng("astigmatism_ratio"),
            shot_noise_prob=prob("shot_noise_prob"),
            dose=rng("dose"),
            detector_noise_prob=prob("detector_noise_prob"),
            detector_sigma=rng("detector_sigma"),
            vignette_prob=prob("vignette_prob"),
            vignette_strength=rng("vignette_strength"),
            gamma_prob=prob("gamma_prob"),
            gamma=rng("gamma"),
            barrel_prob=prob("barrel_prob"),
            barrel_k=rng("barrel_k"),
            charging_prob=prob("charging_prob"),
            charging_rate=rng("charging_rate"),
            charging_gain=rng("charging_gain"),
            drift_jitter_prob=prob("drift_jitter_prob"),
            shear_px=rng("shear_px"),
            jitter_px=rng("jitter_px"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"enabled": self.enabled}
        for stage in EXTENDED_STAGES:
            out[f"{stage}_prob"] = float(getattr(self, f"{stage}_prob"))
        for key in (
            "beam_sigma",
            "astigmatism_ratio",
            "dose",
            "detector_sigma",
            "vignette_strength",
            "gamma",
            "barrel_k",
            "charging_rate",
            "charging_gain",
            "shear_px",
            "jitter_px",
        ):
            out[key] = list(getattr(self, key))
        out["metadata"] = dict(self.metadata)
        return out

    def with_overrides(self, **kwargs: Any) -> "ExtendedDegradationConfig":
        return replace(self, **kwargs)


# --------------------------------------------------------------------------------------
# primitive operations (each takes and returns (H, W, C) float32, no clipping)
# --------------------------------------------------------------------------------------
def apply_beam_blur(
    image: np.ndarray, sigma: float, astigmatism_ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """Anisotropic Gaussian PSF. ``astigmatism_ratio`` stretches sigma along one axis.

    A random axis orientation (0 or 90 degrees) is chosen so the sharper direction is
    not always the same, matching real astigmatism which has an arbitrary axis.
    """
    from scipy.ndimage import gaussian_filter

    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    ratio = max(1.0, float(astigmatism_ratio))
    sig_major = float(sigma) * ratio
    sig_minor = float(sigma)
    if rng.random() < 0.5:
        sig_y, sig_x = sig_major, sig_minor
    else:
        sig_y, sig_x = sig_minor, sig_major
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        out[:, :, c] = gaussian_filter(image[:, :, c], sigma=(sig_y, sig_x), mode="reflect")
    return out


def apply_shot_noise(image: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise. ``I -> Poisson(clip(I,0)*dose)/dose``. Noise ~ 1/sqrt(dose)."""
    if dose <= 0.0:
        return image.astype(np.float32, copy=True)
    lam = np.clip(image, 0.0, None).astype(np.float64) * float(dose)
    counts = rng.poisson(lam).astype(np.float64)
    return (counts / float(dose)).astype(np.float32)


def apply_detector_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian readout noise, independent of signal. No clipping."""
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    return (image + noise).astype(np.float32)


def _radial_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1.0)
    nx = (xx - cx) / max(cx, 1.0)
    r = np.sqrt(nx * nx + ny * ny)
    r = np.clip(r / max(float(r.max()), 1e-6), 0.0, 1.0)
    return r, ny, nx


def apply_vignette(image: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening ``I * (1 - strength * r**2)``."""
    if strength <= 0.0:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    r, _, _ = _radial_grid(h, w)
    mask = (1.0 - float(strength) * r * r).astype(np.float32)
    return (image * mask[:, :, None]).astype(np.float32)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Detector gamma. Operates on the clipped-to-nonnegative signal to stay real."""
    if gamma <= 0.0 or abs(gamma - 1.0) < 1e-6:
        return image.astype(np.float32, copy=True)
    base = np.clip(image, 0.0, None)
    return np.power(base, float(gamma)).astype(np.float32)


def apply_barrel(image: np.ndarray, k: float) -> np.ndarray:
    """Barrel/pincushion distortion via radial remap ``r' = r*(1 + k*r**2)``."""
    if abs(k) < 1e-6:
        return image.astype(np.float32, copy=True)
    from scipy.ndimage import map_coordinates

    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1.0)
    nx = (xx - cx) / max(cx, 1.0)
    r2 = nx * nx + ny * ny
    factor = 1.0 + float(k) * r2
    src_y = cy + (ny * factor) * max(cy, 1.0)
    src_x = cx + (nx * factor) * max(cx, 1.0)
    coords = np.stack([src_y.ravel(), src_x.ravel()], axis=0)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        sampled = map_coordinates(
            image[:, :, c], coords, order=1, mode="reflect"
        ).reshape(h, w)
        out[:, :, c] = sampled
    return out


def apply_charging(
    image: np.ndarray, rate: float, gain: float, rng: np.random.Generator
) -> np.ndarray:
    """Sparse bright horizontal streaks. ``n ~ Poisson(rate * h/100)`` bands."""
    if rate <= 0.0 or gain <= 0.0:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    n = int(rng.poisson(float(rate) * h / 100.0))
    if n <= 0:
        return image.astype(np.float32, copy=True)
    out = image.astype(np.float32, copy=True)
    for _ in range(n):
        row = int(rng.integers(0, h))
        half = int(rng.integers(0, 2))  # 0 -> single row, 1 -> +/-1 band
        g = float(gain) * float(rng.uniform(0.6, 1.0))
        lo = max(0, row - half)
        hi = min(h, row + half + 1)
        out[lo:hi, :, :] += g
    return out


def apply_drift_jitter(
    image: np.ndarray, shear_px: float, jitter_px: float, rng: np.random.Generator
) -> np.ndarray:
    """Progressive row-to-row horizontal shear (drift) plus per-row jitter (vibration)."""
    if shear_px <= 0.0 and jitter_px <= 0.0:
        return image.astype(np.float32, copy=True)
    from scipy.ndimage import map_coordinates

    h, w = image.shape[:2]
    row_idx = np.arange(h, dtype=np.float32)
    drift = (row_idx / max(h - 1, 1)) * float(shear_px)
    jitter = rng.normal(0.0, float(jitter_px), size=h).astype(np.float32) if jitter_px > 0 else 0.0
    dx = drift + jitter  # per-row horizontal shift
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = xx - dx[:, None]
    coords = np.stack([yy.ravel(), src_x.ravel()], axis=0)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        sampled = map_coordinates(image[:, :, c], coords, order=1, mode="reflect").reshape(h, w)
        out[:, :, c] = sampled
    return out


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def apply_extended(
    image: np.ndarray,
    config: ExtendedDegradationConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the enabled extended artifacts in :data:`EXTENDED_STAGES` order.

    Returns the augmented image (same ``(H, W, C)`` shape, unclipped) and a dict of
    the realized parameters for provenance. A dedicated generator, offset from the
    sample seed, keeps these draws independent of the core-degradation realization.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    realized: dict[str, Any] = {}
    if not config.any_active:
        return np.ascontiguousarray(arr), realized

    rng = np.random.default_rng((int(seed) ^ 0x5A17C0DE) % (2**63))

    for stage in EXTENDED_STAGES:
        prob = float(getattr(config, f"{stage}_prob"))
        if prob <= 0.0 or rng.random() >= prob:
            continue
        if stage == "beam_blur":
            sigma = float(rng.uniform(*config.beam_sigma))
            ratio = float(rng.uniform(*config.astigmatism_ratio))
            arr = apply_beam_blur(arr, sigma, ratio, rng)
            realized["beam_blur"] = {"sigma": round(sigma, 4), "astig": round(ratio, 4)}
        elif stage == "shot_noise":
            dose = float(rng.uniform(*config.dose))
            arr = apply_shot_noise(arr, dose, rng)
            realized["shot_noise"] = {"dose": round(dose, 2)}
        elif stage == "detector_noise":
            sigma = float(rng.uniform(*config.detector_sigma))
            arr = apply_detector_noise(arr, sigma, rng)
            realized["detector_noise"] = {"sigma": round(sigma, 5)}
        elif stage == "vignette":
            strength = float(rng.uniform(*config.vignette_strength))
            arr = apply_vignette(arr, strength)
            realized["vignette"] = {"strength": round(strength, 4)}
        elif stage == "gamma":
            g = float(rng.uniform(*config.gamma))
            arr = apply_gamma(arr, g)
            realized["gamma"] = {"gamma": round(g, 4)}
        elif stage == "barrel":
            k = float(rng.uniform(*config.barrel_k))
            arr = apply_barrel(arr, k)
            realized["barrel"] = {"k": round(k, 4)}
        elif stage == "charging":
            rate = float(rng.uniform(*config.charging_rate))
            gain = float(rng.uniform(*config.charging_gain))
            arr = apply_charging(arr, rate, gain, rng)
            realized["charging"] = {"rate": round(rate, 4), "gain": round(gain, 4)}
        elif stage == "drift_jitter":
            shear = float(rng.uniform(*config.shear_px))
            jitter = float(rng.uniform(*config.jitter_px))
            arr = apply_drift_jitter(arr, shear, jitter, rng)
            realized["drift_jitter"] = {"shear": round(shear, 4), "jitter": round(jitter, 4)}

    return np.ascontiguousarray(arr.astype(np.float32)), realized


def describe_extended(config: ExtendedDegradationConfig) -> str:
    """One-line summary for logs."""
    if not config.enabled:
        return "extended=disabled"
    active = [s for s in EXTENDED_STAGES if float(getattr(config, f"{s}_prob")) > 0.0]
    return f"extended=enabled stages={','.join(active) if active else 'none'}"
