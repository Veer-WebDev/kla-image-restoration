#!/usr/bin/env python3
"""Evaluate Drift-Sense localization over a dataset manifest.

Reads a manifest CSV (columns id, reference_path, search_path, gt_x, gt_y as
produced by the Drift-Sense generator) and reports Euclidean pixel error
statistics plus success rates at several pixel thresholds. Optionally splits
the report by whether the localizer flagged the sample as ambiguous (more than
one competing correlation peak), which is the honest signal that a single
degraded Search image may not uniquely determine the location.

    python evaluate.py --manifest data/.../test/manifest.csv [--verify]
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402

from drift_localize import predict  # noqa: E402

THRESHOLDS = (2.0, 5.0, 10.0, 20.0)
DEFAULT_CM_THRESHOLDS = (1.0, 5.0)


def _resolve(path: str, manifest: str) -> str:
    if os.path.exists(path):
        return path
    norm = path.lstrip("./").lstrip(".\\")
    d = os.path.dirname(os.path.abspath(manifest))
    for _ in range(5):
        cand = os.path.normpath(os.path.join(d, norm))
        if os.path.exists(cand):
            return cand
        # also try just the tail (split/reference/file.png) under manifest dir
        tail = os.path.join(d, *path.replace("\\", "/").split("/")[-2:])
        if os.path.exists(tail):
            return tail
        d = os.path.dirname(d)
    return path


def _summary(errs: np.ndarray) -> dict:
    return {
        "n": int(errs.size),
        "mean_px": float(errs.mean()),
        "median_px": float(np.median(errs)),
        "p90_px": float(np.percentile(errs, 90)),
        "max_px": float(errs.max()),
        **{f"success_at_{int(t)}px": float((errs <= t).mean())
           for t in THRESHOLDS},
    }


def _positive_pair_confusion(errs: np.ndarray, tolerance_px: float) -> dict:
    """Summarize localization outcomes at one spatial tolerance.

    Every manifest row is a *positive* reference/search pair with exactly one
    known location. Therefore a conventional four-cell classification confusion
    matrix is not identifiable: there are no negative pairs from which to count
    true negatives or false positives. We report the two meaningful cells
    without inventing negative examples: a hit is a true positive, and a miss
    is a false negative. This is the requested 1--5px CM evidence in the form
    the localization task actually permits.
    """
    tp = int((errs <= tolerance_px).sum())
    fn = int(errs.size - tp)
    return {
        "tolerance_px": float(tolerance_px),
        "true_positive_within_tolerance": tp,
        "false_negative_outside_tolerance": fn,
        "true_negative": None,
        "false_positive": None,
        "recall_or_success_rate": float(tp / max(int(errs.size), 1)),
        "note": ("All manifest rows are positive localization pairs. TN/FP "
                 "are undefined unless a separate negative-pair protocol is used."),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--no-center-tiebreak", action="store_true",
                   help="use raw NCC argmax instead of the official closest-to-"
                        "center tie-break. Useful only for legacy arbitrary-crop "
                        "synthetic labels; do not use for a spec-compliant run.")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="optional Reference-to-Search downscale candidates for "
                        "a robustness study, e.g. 8 8.5 ... 12")
    p.add_argument("--angles", type=float, nargs="+", default=None,
                   help="optional template-angle candidates in degrees for a "
                        "rotation robustness study, e.g. -3 -2 -1 0 1 2 3")
    p.add_argument("--cm-thresholds", type=float, nargs="+",
                   default=DEFAULT_CM_THRESHOLDS,
                   help="localization tolerances for positive-pair confusion "
                        "summaries (default: 1 5)")
    p.add_argument("--json-out", default=None, help="optional path for a JSON report")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    errs, ambig_flags = [], []
    t0 = time.time()
    for r in rows:
        ref = _resolve(r["reference_path"], args.manifest)
        srch = _resolve(r["search_path"], args.manifest)
        predict_kwargs = {"verify": args.verify,
                          "center_tiebreak": not args.no_center_tiebreak}
        if args.scales is not None:
            predict_kwargs["scales"] = tuple(args.scales)
        if args.angles is not None:
            predict_kwargs["angles"] = tuple(args.angles)
        res = predict(ref, srch, **predict_kwargs)
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        errs.append(((res.x - gx) ** 2 + (res.y - gy) ** 2) ** 0.5)
        ambig_flags.append(res.ambiguous)
    dt = time.time() - t0

    errs = np.array(errs)
    ambig = np.array(ambig_flags)
    report = {
        "manifest": os.path.abspath(args.manifest),
        "verify": args.verify,
        "center_tiebreak": not args.no_center_tiebreak,
        "scales": args.scales,
        "angles": args.angles,
        "ms_per_sample": float(dt / max(len(errs), 1) * 1000.0),
        "overall": _summary(errs),
        "positive_pair_confusion": [
            _positive_pair_confusion(errs, t) for t in args.cm_thresholds
        ],
    }
    if ambig.any():
        report["flagged_unique"] = _summary(errs[~ambig]) if (~ambig).any() else None
        report["flagged_ambiguous"] = _summary(errs[ambig])

    o = report["overall"]
    print(f"n={o['n']}  mean={o['mean_px']:.2f}px  median={o['median_px']:.2f}px  "
          f"p90={o['p90_px']:.2f}px  max={o['max_px']:.2f}px")
    for t in THRESHOLDS:
        print(f"  success@{int(t)}px = {o[f'success_at_{int(t)}px']*100:.1f}%")
    print(f"  time/sample = {report['ms_per_sample']:.1f} ms")
    print("  localization confusion summaries (positive pairs only):")
    for cm in report["positive_pair_confusion"]:
        print(f"    @{cm['tolerance_px']:g}px: TP={cm['true_positive_within_tolerance']}  "
              f"FN={cm['false_negative_outside_tolerance']}  "
              f"success={cm['recall_or_success_rate']*100:.1f}%  "
              "TN/FP=undefined (no negative pairs)")
    if "flagged_ambiguous" in report:
        fu, fa = report.get("flagged_unique"), report["flagged_ambiguous"]
        if fu:
            print(f"  unique-peak subset:    n={fu['n']:>4}  "
                  f"success@10px={fu['success_at_10px']*100:.1f}%  median={fu['median_px']:.2f}px")
        print(f"  ambiguous-peak subset: n={fa['n']:>4}  "
              f"success@10px={fa['success_at_10px']*100:.1f}%  median={fa['median_px']:.2f}px")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
