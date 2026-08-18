from __future__ import annotations

import numpy as np

from kla_restore.extended_degradation import (
    EXTENDED_STAGES,
    ExtendedDegradationConfig,
    apply_barrel,
    apply_beam_blur,
    apply_charging,
    apply_detector_noise,
    apply_drift_jitter,
    apply_extended,
    apply_gamma,
    apply_shot_noise,
    apply_vignette,
)


def _all_on(**overrides) -> ExtendedDegradationConfig:
    base = dict(
        enabled=True,
        beam_blur_prob=1.0,
        shot_noise_prob=1.0,
        detector_noise_prob=1.0,
        vignette_prob=1.0,
        gamma_prob=1.0,
        barrel_prob=1.0,
        charging_prob=1.0,
        drift_jitter_prob=1.0,
    )
    base.update(overrides)
    return ExtendedDegradationConfig.from_dict(base)


def test_default_config_is_noop() -> None:
    cfg = ExtendedDegradationConfig()
    assert not cfg.enabled
    assert not cfg.any_active
    img = np.random.default_rng(0).random((32, 32, 1), dtype=np.float32)
    out, params = apply_extended(img, cfg, seed=1)
    assert np.array_equal(out, img)
    assert params == {}


def test_enabled_but_all_zero_prob_is_noop() -> None:
    cfg = ExtendedDegradationConfig(enabled=True)
    assert not cfg.any_active
    img = np.full((16, 16, 1), 0.5, dtype=np.float32)
    out, params = apply_extended(img, cfg, seed=7)
    assert np.array_equal(out, img)
    assert params == {}


def test_same_seed_reproduces_exact_output() -> None:
    cfg = _all_on()
    img = np.random.default_rng(3).random((48, 48, 1), dtype=np.float32)
    a, pa = apply_extended(img, cfg, seed=42)
    b, pb = apply_extended(img, cfg, seed=42)
    assert np.array_equal(a, b)
    assert pa == pb
    assert a.shape == img.shape
    assert a.dtype == np.float32


def test_different_seed_changes_output() -> None:
    cfg = _all_on()
    img = np.full((48, 48, 1), 0.5, dtype=np.float32)
    a, _ = apply_extended(img, cfg, seed=1)
    b, _ = apply_extended(img, cfg, seed=2)
    assert not np.array_equal(a, b)


def test_all_stages_recorded_when_forced() -> None:
    cfg = _all_on()
    img = np.random.default_rng(5).random((64, 64, 1), dtype=np.float32)
    _, params = apply_extended(img, cfg, seed=9)
    # gamma with ratio exactly 1.0 or barrel k~0 could no-op, but ranges exclude that.
    for stage in EXTENDED_STAGES:
        assert stage in params, f"missing {stage}"


def test_shape_preserved_for_3_channel() -> None:
    cfg = _all_on()
    img = np.random.default_rng(6).random((40, 40, 3), dtype=np.float32)
    out, _ = apply_extended(img, cfg, seed=11)
    assert out.shape == (40, 40, 3)


def test_shot_noise_relative_variance_scales_with_dose() -> None:
    rng = np.random.default_rng(0)
    img = np.full((256, 256, 1), 0.5, dtype=np.float32)
    low = apply_shot_noise(img, dose=100.0, rng=np.random.default_rng(1))
    high = apply_shot_noise(img, dose=4000.0, rng=np.random.default_rng(1))
    assert low.std() > high.std()


def test_primitives_do_not_return_nan() -> None:
    rng = np.random.default_rng(0)
    img = np.random.default_rng(0).random((32, 32, 1), dtype=np.float32)
    for fn in (
        lambda: apply_beam_blur(img, 1.0, 2.0, rng),
        lambda: apply_shot_noise(img, 500.0, rng),
        lambda: apply_detector_noise(img, 0.02, rng),
        lambda: apply_vignette(img, 0.3),
        lambda: apply_gamma(img, 1.5),
        lambda: apply_barrel(img, 0.1),
        lambda: apply_charging(img, 0.1, 0.4, rng),
        lambda: apply_drift_jitter(img, 3.0, 1.0, rng),
    ):
        out = fn()
        assert np.isfinite(out).all()
        assert out.shape == img.shape


def test_from_dict_rejects_unknown_keys() -> None:
    try:
        ExtendedDegradationConfig.from_dict({"nope": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown key")


def test_roundtrip_to_from_dict() -> None:
    cfg = _all_on(beam_sigma=[0.5, 0.9])
    again = ExtendedDegradationConfig.from_dict(cfg.to_dict())
    assert again.beam_sigma == (0.5, 0.9)
    assert again.any_active
