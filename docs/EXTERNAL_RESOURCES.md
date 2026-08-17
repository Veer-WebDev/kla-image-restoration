# External Resources and Data Provenance

This is the authoritative disclosure log for external code, data and weights
used in this repository.

## Runtime dependencies

| Resource | Role | Licence / source | Status |
|---|---|---|---|
| NumPy | Array math | BSD. https://numpy.org/ | Used, declared in `requirements.txt`. |
| OpenCV (opencv-python-headless) | Image IO, `matchTemplate`, resize | Apache-2.0. https://opencv.org/ | Used, declared in `requirements.txt`. |

The solver has no deep-learning dependency and requires no network access or
GPU at inference time.

## Drift-Sense synthetic data generator

- **Name:** Drift-Sense Synthetic Dataset Generator
- **URL:** https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data
- **Purpose:** generates synthetic Reference/Search image pairs with ground-truth
  centre coordinates for the Applied Materials Drift-Sense localization task,
  which is exactly the task this repository solves.
- **How it is used here:** the generator is run **locally** to produce seeded
  train/val/test splits used only to *measure* the solver. The generator code is
  **not vendored or redistributed** in this repository. No generated pixels are
  committed; only aggregate metrics (in `results/experiments.csv` and
  `results/localize_*.json`) are.
- **Licence caution:** at the inspected revision the Space exposed no explicit
  `LICENSE` file. Because the solver here is a clean, independent reimplementation
  of standard normalized cross-correlation (OpenCV `matchTemplate`) and contains
  **no code copied** from the Space, its licensing status does not encumber this
  repository. The generator is treated purely as an external measurement fixture.

## Honesty notes

- All committed metrics are from **synthetic** Drift-Sense data and are labelled
  synthetic. No metric is claimed as an official KLA/AMAT leaderboard result.
- No parameter is tuned against hidden test data. Splits are seeded and disjoint.
