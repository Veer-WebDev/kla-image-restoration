#!/usr/bin/env python3
"""Standalone Drift-Sense localization CLI (official submission interface).

Accepts a Reference and a Search image path and prints the predicted centre
"x,y" (pixels) in the Search image, exactly as the Applied Materials Drift-Sense
scoring harness expects. Imports only numpy and OpenCV; no deep-learning
dependency, no network access.

    python infer.py --reference ref.png --search search.png
    # -> 512.34,488.10
"""

import argparse
import sys

# Allow running as a plain script (python infer.py ...) without installation.
sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "src"))

from drift_localize import predict  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="reference image path")
    p.add_argument("--search", required=True, help="search image path")
    p.add_argument("--verify", action="store_true",
                   help="enable fine-resolution re-verification stage")
    args = p.parse_args()

    result = predict(args.reference, args.search, verify=args.verify)
    print(f"{result.x:.2f},{result.y:.2f}")


if __name__ == "__main__":
    main()
