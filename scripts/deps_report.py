"""Report which optional dependencies exist in the active interpreter.

Internal dev tool. Evaluation code must never require a package this reports as
MISSING, so this is the authoritative list before writing anything that imports.
"""
from __future__ import annotations

import importlib
import sys

MODULES = (
    "numpy",
    "torch",
    "PIL",
    "yaml",
    "skimage",
    "scipy",
    "matplotlib",
    "pytest",
    "lpips",
)


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reporting tool
            print(f"  {name:12s} MISSING ({type(exc).__name__})")
            continue
        print(f"  {name:12s} {getattr(module, '__version__', 'unknown')}")
    try:
        import torch

        print(f"  cuda available: {torch.cuda.is_available()}")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
