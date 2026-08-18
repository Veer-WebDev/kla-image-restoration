from __future__ import annotations

import numpy as np
import torch

from kla_restore.metrics import compute_metrics, mae, psnr, ssim, ssim_torch


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


def test_ssim_torch_stays_bounded_in_fp16_on_low_variance_inputs() -> None:
    """Regression: high-mean/low-variance patches used to make the variance
    estimate cancel to a negative value under fp16, flip the denominator sign
    and blow SSIM up to huge magnitudes (loss -> large negative, divergence).
    SSIM must stay finite and within [-1, 1] in half precision."""
    torch.manual_seed(0)
    pred = (torch.rand(4, 1, 128, 128) * 0.02 + 0.9).half().clamp(0, 1)
    target = (torch.rand(4, 1, 128, 128) * 0.02 + 0.9).half().clamp(0, 1)

    value = float(ssim_torch(pred, target))

    assert np.isfinite(value)
    assert -1.0001 <= value <= 1.0001


def test_ssim_torch_bounded_across_precisions_and_ranges() -> None:
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.float16):
        for lo, hi in ((0.0, 1.0), (0.9, 0.92), (0.0, 0.05)):
            p = (torch.rand(2, 1, 96, 96) * (hi - lo) + lo).to(dtype).clamp(0, 1)
            t = (torch.rand(2, 1, 96, 96) * (hi - lo) + lo).to(dtype).clamp(0, 1)
            v = float(ssim_torch(p, t))
            assert np.isfinite(v)
            assert -1.0001 <= v <= 1.0001
