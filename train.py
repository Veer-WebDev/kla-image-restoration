#!/usr/bin/env python3
"""Entry point for training. All logic lives in ``kla_restore.train``.

Examples
--------
    python train.py --gt-dir data/GT --noisy-dir data/NoisyLR
    python train.py --epochs 2 --set data.samples_per_image=2   # quick check
    python train.py --resume runs/baseline_residual_unet/last.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kla_restore.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
