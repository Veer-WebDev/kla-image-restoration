#!/usr/bin/env python3
"""Experimental feature-matching baseline for Drift-Sense.

This script is not part of deployment. It measures whether an invariant local
feature method can improve on NCC before adding complexity. The Reference is
resized over the same 9--11x candidate scale range, SIFT descriptors are matched
to the Search, and RANSAC estimates a homography. The best inlier-ratio candidate
returns the transformed template center.

It should be retained only if it improves a fixed, representative synthetic
holdout. Output is explicitly marked synthetic.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import _resolve  # noqa: E402

SCALES = (9.0, 9.5, 10.0, 10.5, 11.0)


def _predict(reference_path: str, search_path: str) -> tuple[float, float, int, float]:
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise ValueError(f"could not read {reference_path} or {search_path}")
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.01)
    target_kp, target_desc = sift.detectAndCompute(search, None)
    if target_desc is None:
        return search.shape[1] / 2, search.shape[0] / 2, 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = None  # (inlier count, ratio, x, y)
    for scale in SCALES:
        template = cv2.resize(reference, (round(reference.shape[1] / scale),
                                          round(reference.shape[0] / scale)),
                              interpolation=cv2.INTER_AREA)
        source_kp, source_desc = sift.detectAndCompute(template, None)
        if source_desc is None or len(source_kp) < 4:
            continue
        pairs = matcher.knnMatch(source_desc, target_desc, k=2)
        good = [a for a, b in pairs if a.distance < 0.70 * b.distance]
        if len(good) < 4:
            continue
        source = np.float32([source_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            continue
        center = np.float32([[[template.shape[1] / 2, template.shape[0] / 2]]])
        mapped = cv2.perspectiveTransform(center, H)[0, 0]
        if not (0 <= mapped[0] < search.shape[1] and 0 <= mapped[1] < search.shape[0]):
            continue
        inliers = int(mask.ravel().sum())
        candidate = (inliers, inliers / len(good), float(mapped[0]), float(mapped[1]))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return search.shape[1] / 2, search.shape[0] / 2, 0, 0.0
    return best[2], best[3], best[0], best[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.manifest, newline="")))[:args.limit]
    errors, inliers, ratios = [], [], []
    t0 = time.perf_counter()
    for row in rows:
        x, y, n, ratio = _predict(_resolve(row["reference_path"], args.manifest),
                                  _resolve(row["search_path"], args.manifest))
        errors.append(float(np.hypot(x - float(row["gt_x"]), y - float(row["gt_y"]))))
        inliers.append(n)
        ratios.append(ratio)
    elapsed = time.perf_counter() - t0
    e = np.asarray(errors)
    report = {"synthetic": True, "method": "SIFT + BF ratio match + RANSAC homography",
              "n": len(rows), "median_px": float(np.median(e)), "mean_px": float(e.mean()),
              "success_at_5px": float((e <= 5).mean()), "success_at_10px": float((e <= 10).mean()),
              "ms_per_sample": elapsed / max(len(rows), 1) * 1000,
              "median_inliers": float(np.median(inliers)),
              "median_inlier_ratio": float(np.median(ratios))}
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
