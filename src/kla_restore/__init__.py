"""KLA semiconductor image restoration.

Reproducible restoration pipeline for the SEMICON India Hackathon 2026 KLA problem
statement: NoisyLR -> bicubic upsample -> residual U-Net -> clamp -> restored image.

The public API is intentionally small and stable so that ``train.py``, ``inference.py``
and ``evaluate.py`` never reach into private helpers.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "CHECKPOINT_FORMAT_VERSION",
]

# Bumped whenever the on-disk checkpoint layout changes in a backwards-incompatible way.
CHECKPOINT_FORMAT_VERSION = 2
