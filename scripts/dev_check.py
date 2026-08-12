"""Developer smoke check for the core library.

Not a deliverable and not a substitute for tests/: this is the fast, dependency-light
gate used during development to confirm the pieces still fit together.

Usage
-----
    python scripts/dev_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kla_restore.checkpoint import CheckpointMeta, load_model, save_checkpoint  # noqa: E402
from kla_restore.degradation import (  # noqa: E402
    ORDER_PERMUTATIONS,
    DegradationConfig,
    degrade,
    order_matrix,
    sample_seed,
)
from kla_restore.metrics import build_loss, compute_metrics  # noqa: E402
from kla_restore.model import ModelConfig, build_model, model_summary  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" | {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 74)
    print("kla_restore developer check")
    print("=" * 74)

    check("six degradation orders", len(ORDER_PERMUTATIONS) == 6, str(len(ORDER_PERMUTATIONS)))

    cfg = DegradationConfig()
    gt = np.random.default_rng(0).random((64, 64, 1)).astype("float32")

    seed_a = sample_seed(42, "img001", 0, 0)
    noisy, params = degrade(gt, cfg, seed_a)
    check(
        "downsample factor honoured",
        noisy.shape[0] == 64 // params.scale,
        f"{noisy.shape} scale={params.scale} order={'->'.join(params.order)}",
    )

    # Clipping must never be applied: saturate the GT so noise provably leaves [0, 1].
    saturated = np.ones((32, 32, 1), dtype="float32")
    noisy_sat, sat_params = degrade(
        saturated, cfg.with_overrides(fixed_order=("gaussian", "speckle", "downsample")), 7
    )
    check(
        "noisy is never clipped",
        float(noisy_sat.max()) > 1.0,
        f"max={noisy_sat.max():.4f} sigma_g={sat_params.gaussian_sigma:.4f}",
    )

    noisy_again, _ = degrade(gt, cfg, seed_a)
    check("same seed reproduces sample", np.array_equal(noisy, noisy_again))

    noisy_e1, _ = degrade(gt, cfg, sample_seed(42, "img001", 0, 1))
    check("epoch changes the sample", not np.array_equal(noisy, noisy_e1))

    noisy_i1, _ = degrade(gt, cfg, sample_seed(42, "img001", 1, 0))
    check("sample index changes the sample", not np.array_equal(noisy, noisy_i1))

    matrix = order_matrix(gt, cfg, 123)
    distinct = len({a.tobytes() for a in matrix.values()})
    check("order matrix has 6 entries", len(matrix) == 6)
    check("orderings differ numerically", distinct >= 4, f"{distinct}/6 distinct")

    model = build_model(ModelConfig(base_channels=32, depth=4))
    summary = model_summary(model)
    check(
        "model builds",
        summary["params_total"] > 0,
        f"{summary['params_millions']}M params, {summary['fp32_size_mb']}MB fp32",
    )

    x = torch.from_numpy(noisy.transpose(2, 0, 1))[None]
    with torch.inference_mode():
        y_scale = model(x, scale=params.scale)
    check(
        "scale contract sizes output",
        tuple(y_scale.shape[-2:]) == (x.shape[-2] * params.scale, x.shape[-1] * params.scale),
        f"{tuple(x.shape)} -> {tuple(y_scale.shape)}",
    )

    with torch.inference_mode():
        y_odd = model(torch.rand(1, 1, 37, 53), scale=2)
    check("odd input sizes work", tuple(y_odd.shape) == (1, 1, 74, 106), str(tuple(y_odd.shape)))

    with torch.inference_mode():
        eval_parts = model(x, target_size=(64, 64), return_parts=True)
    residual_max = float(eval_parts["residual"].abs().max())
    check(
        "near-identity init stays close to bicubic",
        residual_max < 0.05,
        f"max|residual|={residual_max:.6f}",
    )

    # Gradient path: outside inference_mode, and with clamping disabled.
    gt_t = torch.from_numpy(gt.transpose(2, 0, 1))[None]
    train_parts = model(x, target_size=(64, 64), clamp=False, return_parts=True)
    loss_fn = build_loss()
    total, terms = loss_fn(train_parts["restored"], gt_t)
    check("loss is finite", bool(torch.isfinite(total)), f"{float(total):.6f} {terms}")

    model.zero_grad(set_to_none=True)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    nonzero = sum(1 for g in grads if float(g.abs().sum()) > 0)
    check(
        "gradients flow",
        len(grads) > 0 and nonzero > 0,
        f"{nonzero}/{len(grads)} tensors with nonzero grad",
    )

    pred_np = eval_parts["restored"][0].permute(1, 2, 0).numpy()
    metrics = compute_metrics(pred_np, gt)
    check(
        "metrics computed",
        all(np.isfinite(v) for v in metrics.values()),
        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
    )

    ckpt_path = ROOT / "runs" / "_devcheck" / "model.pth"
    meta = CheckpointMeta(
        experiment_id="devcheck",
        model_config=model.config.to_dict(),
        degradation_config=cfg.to_dict(),
        inference_scale=2,
    )
    save_checkpoint(ckpt_path, model, meta)
    reloaded, meta2 = load_model(ckpt_path)
    same = all(
        torch.equal(a.detach(), b.detach())
        for a, b in zip(model.state_dict().values(), reloaded.state_dict().values())
    )
    check(
        "checkpoint roundtrip is exact",
        same and meta2.experiment_id == "devcheck" and meta2.inference_scale == 2,
        f"format_version={meta2.format_version}",
    )

    print("=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
