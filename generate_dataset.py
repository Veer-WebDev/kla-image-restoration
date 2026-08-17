#!/usr/bin/env python3
"""Generate a synthetic Drift-Sense dataset (reference/search pairs + manifest).

Produces the grayscale image pairs required by the Applied Materials
"Drift-Sense" task using src/drift_localize/generator.py, and writes a
manifest CSV compatible with evaluate.py.

    python generate_dataset.py --out data/mydata --n 30 --seed 31337
    python generate_dataset.py --out data/noisy --n 30 --search-speckle 0.3

The generator uses only publicly known DRAM/FinFET structural characteristics
and literature-backed SEM noise models (see generator.py docstring). No
proprietary fab data is involved.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from drift_localize import generator  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, default=30, help="number of pairs")
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--arch", choices=["dram", "finfet", "mixed"], default="mixed",
                   help="structure style; 'mixed' alternates dram/finfet")
    p.add_argument("--search-speckle", type=float, default=0.0,
                   help="multiplicative speckle sigma on the search image "
                        "(robustness stress test; FAQ: search is noisier in test data)")
    p.add_argument("--search-readout", type=float, default=5.0,
                   help="additive readout-noise sigma on the search image")
    args = p.parse_args()

    ref_dir = os.path.join(args.out, "reference")
    srch_dir = os.path.join(args.out, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(srch_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        if args.arch == "mixed":
            arch = "dram" if i % 2 == 0 else "finfet"
        else:
            arch = args.arch
        s = generator.generate_sample(
            arch, rng, search_speckle_sigma=args.search_speckle,
            search_readout_sigma=args.search_readout)
        rid = f"{i:05d}"
        ref_path = os.path.join("reference", f"{rid}.png")
        srch_path = os.path.join("search", f"{rid}.png")
        cv2.imwrite(os.path.join(args.out, ref_path), s["reference_img"])
        cv2.imwrite(os.path.join(args.out, srch_path), s["search_img"])
        rows.append({"id": rid, "architecture": arch,
                     "reference_path": ref_path.replace("\\", "/"),
                     "search_path": srch_path.replace("\\", "/"),
                     "gt_x": f"{s['gt_x']:.4f}", "gt_y": f"{s['gt_y']:.4f}"})

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(generator.MANIFEST_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.n} pairs to {args.out} (seed={args.seed}, "
          f"speckle={args.search_speckle}); manifest {manifest}")


if __name__ == "__main__":
    main()
