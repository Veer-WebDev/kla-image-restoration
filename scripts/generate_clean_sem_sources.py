#!/usr/bin/env python3
"""Generate clean first-party semiconductor-inspired source images.

This generator is deliberately independent of the unlicensed Drift-Sense Space.
It synthesizes grayscale line-space, contact-array, routing-trace and mixed
motifs from simple public semiconductor layout characteristics.  It generates
clean source images only.  ``materialize_restoration_data.py`` is solely
responsible for the permitted Gaussian, speckle and downsampling degradations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kla_restore.utils import save_image_float  # noqa: E402
from scripts.make_fixtures import PATTERNS, make_gt  # noqa: E402


def _mixed_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Combine independently parameterized first-party structural motifs."""
    kinds = ("grid", "lines", "traces", "contacts")
    base = np.full((size, size, 1), 0.10, dtype=np.float32)
    for _ in range(3):
        component = make_gt(size, kinds[int(rng.integers(0, len(kinds)))], rng)
        opacity = float(rng.uniform(0.28, 0.60))
        base = np.maximum(base, component * opacity)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def generate_sources(*, out_dir: Path, count: int, size: int, seed: int) -> dict[str, object]:
    """Generate a deterministic clean-source corpus and provenance manifest."""
    if count < 3:
        raise ValueError("count must be at least 3 to permit source-disjoint splits")
    if size < 64:
        raise ValueError("size must be at least 64")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    kinds = tuple(PATTERNS) + ("mixed",)
    rows = []
    for index in range(int(count)):
        kind = kinds[index % len(kinds)]
        image = _mixed_pattern(size, rng) if kind == "mixed" else make_gt(size, kind, rng)
        filename = f"source_{index:04d}_{kind}.png"
        path = out_dir / filename
        save_image_float(path, image, bit_depth=8)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"source_file": filename, "source_sha256": digest, "motif": kind})
    summary: dict[str, object] = {
        "synthetic": True,
        "generator": "scripts/generate_clean_sem_sources.py",
        "seed": int(seed),
        "count": int(count),
        "size": int(size),
        "motifs": list(kinds),
        "sources": rows,
        "warning": "First-party synthetic clean-source images only. They are not official KLA data.",
    }
    (out_dir / "source_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)
    print(json.dumps(generate_sources(out_dir=args.out, count=args.count, size=args.size, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
