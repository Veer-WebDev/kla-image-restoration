"""Installed-package entry points.

The evaluator-facing interfaces remain the repository-root ``train.py`` and
``inference.py`` scripts.  This small adapter makes the declared ``kla-train``
console command resolve to the identical training implementation after an
editable or wheel installation.
"""

from __future__ import annotations

from .train import main


def train_main() -> int:
    """Run the training command used by both the package and root script."""
    return main()
