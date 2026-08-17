#!/usr/bin/env python3
"""Measure confidence precision--recall curves across synthetic search noise.

The FAQ asks teams to make the Search noisier than the Reference and the slides
ask for precision--recall versus injected noise. This script uses the tracked,
seeded generator and a disjoint calibration/test split at each noise level:

1. Generate ``--calibration-n`` pairs to choose the NCC score threshold with
   best F1 at ``--tolerance-px``.
2. Generate a different ``--test-n`` set and report the threshold's precision,
   recall and acceptance rate.
3. Write JSON plus a dependency-free SVG PR plot.

The reported precision is *accepted localization correctness*, not object
classification: accepted predictions within tolerance are TP; accepted misses
are FP; rejected within-tolerance predictions are FN; rejected misses are TN.
This makes a confidence threshold meaningful despite every input being a
positive reference/search pair. Results are synthetic and must not be stated as
KLA/AMAT test performance.

Example:
  python analysis/noise_sweep.py --out results/noise_sweep --test-n 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_localize import generator, predict  # noqa: E402


COLORS = ("#147d9b", "#f0ad2c", "#4f8126", "#7e57c2", "#bf3f6b")


def _measure(n: int, seed: int, speckle: float) -> list[dict]:
    """Generate and localize a reproducible synthetic set without retaining pixels."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="drift-sense-noise-") as temp:
        ref_path = os.path.join(temp, "reference.png")
        search_path = os.path.join(temp, "search.png")
        for index in range(n):
            arch = "dram" if index % 2 == 0 else "finfet"
            sample = generator.generate_sample(
                arch, rng, search_speckle_sigma=speckle,
            )
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])
            result = predict(ref_path, search_path)
            error = float(np.hypot(result.x - sample["gt_x"],
                                   result.y - sample["gt_y"]))
            rows.append({"score": float(result.score), "error_px": error,
                         "ambiguous": bool(result.ambiguous)})
    return rows


def _confusion(rows: list[dict], threshold: float, tolerance_px: float) -> dict:
    accepted = [r for r in rows if r["score"] >= threshold]
    rejected = [r for r in rows if r["score"] < threshold]
    tp = sum(r["error_px"] <= tolerance_px for r in accepted)
    fp = len(accepted) - tp
    fn = sum(r["error_px"] <= tolerance_px for r in rejected)
    tn = len(rejected) - fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(threshold), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn), "precision": float(precision),
            "recall": float(recall), "acceptance_rate": len(accepted) / len(rows)}


def _curve(rows: list[dict], tolerance_px: float) -> list[dict]:
    # Endpoints make the curve interpretable: accept all and reject all.
    scores = sorted({float(r["score"]) for r in rows})
    cutoffs = [scores[0] - 1e-6, *scores, scores[-1] + 1e-6]
    return [_confusion(rows, c, tolerance_px) for c in cutoffs]


def _best_f1(curve: list[dict]) -> dict:
    def key(point: dict) -> tuple[float, float, float]:
        p, r = point["precision"], point["recall"]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        # Prefer more recall, then lower cutoff on equal F1.
        return f1, r, -point["threshold"]
    return max(curve, key=key)


def _svg(report: dict, destination: Path) -> None:
    """Render an intentionally portable SVG PR chart, avoiding a plotting dep."""
    width, height = 840, 600
    left, top, plot_w, plot_h = 105, 65, 650, 420

    def px(recall: float) -> float:
        return left + np.clip(recall, 0, 1) * plot_w

    def py(precision: float) -> float:
        return top + (1 - np.clip(precision, 0, 1)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}.small{font-size:13px}.axis{stroke:#283746;stroke-width:1.5}.grid{stroke:#d8dee4;stroke-width:1}</style>',
        '<text x="105" y="32" font-size="21" font-weight="700">Precision--Recall vs. injected Search speckle</text>',
        '<text x="105" y="52" class="small">Synthetic held-out sets. Confidence = NCC peak score. Tolerance shown in report.</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x, y = px(value), py(value)
        parts.extend([
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>',
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>',
            f'<text class="small" x="{x - 9:.1f}" y="{top + plot_h + 23}">{value:.1f}</text>',
            f'<text class="small" x="{left - 35}" y="{y + 4:.1f}">{value:.1f}</text>',
        ])
    parts.extend([
        f'<text x="{left + plot_w / 2 - 32:.1f}" y="{top + plot_h + 53}" font-size="15">Recall</text>',
        f'<text transform="translate(28 {top + plot_h / 2 + 32:.1f}) rotate(-90)" font-size="15">Precision</text>',
    ])
    for index, level in enumerate(report["levels"]):
        color = COLORS[index % len(COLORS)]
        curve = level["calibration_pr_curve"]
        points = " ".join(f"{px(p['recall']):.1f},{py(p['precision']):.1f}" for p in curve)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        chosen = level["test_at_calibrated_threshold"]
        parts.append(f'<circle cx="{px(chosen["recall"]):.1f}" cy="{py(chosen["precision"]):.1f}" r="5" fill="{color}"/>')
        label_y = 105 + index * 25
        parts.append(f'<line x1="{left + plot_w - 190}" y1="{label_y - 5}" x2="{left + plot_w - 170}" y2="{label_y - 5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{left + plot_w - 163}" y="{label_y}">speckle σ={level["speckle_sigma"]:g}</text>')
    parts.append('</svg>')
    destination.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/noise_sweep", help="output prefix or directory")
    parser.add_argument("--levels", type=float, nargs="+", default=(0.0, 0.3, 0.6))
    parser.add_argument("--calibration-n", type=int, default=30)
    parser.add_argument("--test-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    args = parser.parse_args()
    if args.calibration_n < 1 or args.test_n < 1:
        parser.error("--calibration-n and --test-n must be positive")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"synthetic": True, "seed": args.seed, "tolerance_px": args.tolerance_px,
              "calibration_n": args.calibration_n, "test_n": args.test_n,
              "protocol": ("Thresholds maximize F1 on a separate generated calibration "
                           "set at each noise level; points in the plot are calibration PR "
                           "curves and circles are held-out test outcomes."),
              "levels": []}
    for index, speckle in enumerate(args.levels):
        print(f"[{index + 1}/{len(args.levels)}] speckle sigma={speckle:g}: calibration")
        calibration = _measure(args.calibration_n, args.seed + index * 1000, speckle)
        curve = _curve(calibration, args.tolerance_px)
        selected = _best_f1(curve)
        print(f"[{index + 1}/{len(args.levels)}] speckle sigma={speckle:g}: held-out test")
        test = _measure(args.test_n, args.seed + 100_000 + index * 1000, speckle)
        held_out = _confusion(test, selected["threshold"], args.tolerance_px)
        report["levels"].append({
            "speckle_sigma": float(speckle),
            "calibration_pr_curve": curve,
            "calibration_selected_threshold": selected,
            "test_at_calibrated_threshold": held_out,
        })
        print(f"  threshold={selected['threshold']:.3f} | test precision={held_out['precision']*100:.1f}% "
              f"recall={held_out['recall']*100:.1f}% accept={held_out['acceptance_rate']*100:.1f}%")

    json_path = output.with_suffix(".json")
    svg_path = output.with_suffix(".svg")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _svg(report, svg_path)
    print(f"wrote {json_path} and {svg_path}")


if __name__ == "__main__":
    main()
