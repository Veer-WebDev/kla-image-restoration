"""Drift-Sense reference localization.

Task (Applied Materials "Drift-Sense", SEMICON India Hackathon 2026): given a
high-resolution Reference image (1000x1000 px at 1 nm/px, a 1 um field of view)
and a degraded, drifted Search image (1000x1000 px at 10 nm/px, a 10 um field
of view), predict the (x, y) pixel coordinates in the Search image where the
Reference's field of view is centred. The scoring metric is Euclidean pixel
error of that predicted centre.

This package is deliberately classical: normalized cross-correlation template
matching with an optional fine-resolution re-verification stage. See
docs/submission for why a heavier learned model is not justified on this task.
"""

from drift_localize.matcher import predict, LocalizeResult

__all__ = ["predict", "LocalizeResult"]
