"""Curate clean, artifact-free SEM structures from the Drift-Sense dataset drop
into a source-image directory usable by materialize_restoration_data.py.

Only clean GT-quality images are kept: gallery reference crops, full reference
patches, un-annotated search ground-truth renders, and composite layer-stack
grayscales. Charts, QR codes, annotated overlays, pre-degraded artifact demos,
montages and tiny polygon thumbnails are excluded.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def curate(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    keep: list[Path] = []
    for p in sorted(src.glob("*.png")):
        n = p.name.lower()
        if "annotated" in n or "qr" in n:
            continue
        if n.startswith(
            (
                "ap_vs",
                "pr_curves",
                "sample_family",
                "noise_",
                "drift_",
                "distort_",
                "polygon_",
                "aliasing_",
                "hf_space",
            )
        ):
            continue
        if n.endswith("_ref.png") or n.endswith("_reference.png") or n.endswith("_search_gt.png"):
            keep.append(p)
    layers = src / "layers"
    if layers.is_dir():
        keep.extend(sorted(layers.glob("*_composite_grayscale.png")))
    copied: list[str] = []
    for p in keep:
        shutil.copy2(p, dst / p.name)
        copied.append(p.name)
    return copied


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\Administrator\Downloads\dataset")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/sem_sources")
    names = curate(src, dst)
    print(f"copied {len(names)} clean SEM sources to {dst}")
    for n in names:
        print(" ", n)
