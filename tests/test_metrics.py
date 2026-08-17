from __future__ import annotations

import numpy as np

from kla_restore.metrics import compute_metrics, mae, psnr, ssim


def test_perfect_restoration_has_zero_mae_maximum_psnr_and_unit_ssim() -> None:
    target = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8, 1)

    assert mae(target, target) == 0.0
    assert np.isinf(psnr(target, target))
    assert ssim(target, target) == 1.0


def test_metrics_penalize_nonidentical_restoration() -> None:
    target = np.zeros((16, 16, 1), dtype=np.float32)
    prediction = np.full_like(target, 0.25)
    result = compute_metrics(prediction, target)

    assert result["mae"] == 0.25
    assert np.isfinite(result["psnr"])
    assert 0.0 <= result["ssim"] < 1.0
    assert "lpips" not in result
