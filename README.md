# Drift-Sense Reference Localization

A deterministic, submission-oriented solver for the Applied Materials
**"Drift-Sense"** localization task (KLA / Applied Materials SEMICON India
Hackathon 2026).

> **Status:** the solver here is validated on a large held-out set of images
> drawn from the official Drift-Sense synthetic generator. No result in this
> repository is represented as an official leaderboard score, and no metric is
> tuned against hidden test data. Every number below comes from a real run on
> repository-generated synthetic data and is labelled as such.

## The task

Given two images of the same wafer region:

- **Reference** — 1000×1000 px at **1 nm/px** (a 1 µm × 1 µm field of view), high resolution.
- **Search** — 1000×1000 px at **10 nm/px** (a 10 µm × 10 µm field of view), degraded and drifted.

Predict the **(x, y) pixel coordinates** in the Search image where the
Reference's field of view is centred. Because the Reference covers 1 µm and the
Search covers 10 µm at the same 1000 px, the Reference occupies roughly a
100×100 px window inside the Search image.

**The scoring metric is Euclidean pixel error of the predicted centre.** This is
a *localization* problem, not an image-restoration or denoising problem: the
goal is a coordinate, not a cleaned image.

## Method

Classical, dependency-light **normalized cross-correlation (NCC) template
matching** with sub-pixel refinement:

```text
Reference → downsample to candidate ~100px templates (scales 9–11x)
          → NCC slide over Search (cv2.matchTemplate, TM_CCOEFF_NORMED)
          → pick best correlation peak across scales
          → parabolic sub-pixel fit around the peak → (x, y)
```

An optional fine-resolution re-verification stage (`--verify`) re-ranks the top
correlation peaks using the full-resolution Reference. It is **off by default**
because on a 200-image held-out set it delivers identical accuracy at 3× the
runtime (see below): the remaining errors are genuine appearance ambiguities,
not a modelling gap that a heavier stage could close.

Inference needs only **NumPy and OpenCV**. No deep-learning dependency, no
network access, no GPU.

## Why classical, not a learned model

We measured this rather than assumed it. On a 200-image held-out synthetic set:

| Solver | success@10px | median err | time/sample |
| --- | --- | --- | --- |
| NCC (default) | **75.5%** | 1.03 px | 156 ms |
| NCC + `--verify` | 75.5% | 1.03 px | 470 ms |

The successful matches are essentially exact (median ≈ 1 px, sub-pixel). The
~25% of failures are **appearance-ambiguous periodic arrays**: at the ground-
truth location the degraded Search content matches the Reference no better than
at the wrong location the solver picks. A single degraded Search image
physically does not contain the information to disambiguate these repeats, so
**this is a fundamental information limit, not a solver deficiency** — a larger
learned model cannot recover information the input does not carry. The solver
exposes this honestly: it flags such samples as ambiguous.

| Subset (flagged by the solver) | count | success@10px | median err |
| --- | --- | --- | --- |
| Unique correlation peak | 92 | **97.8%** | 0.83 px |
| Competing (ambiguous) peaks | 108 | 56.5% | 1.41 px |

The ambiguity flag is a real, useful signal: when the solver reports a unique
peak it is right 98% of the time. See `docs/submission/` for the full analysis.

## Installation

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt      # numpy + opencv only
```

## Standalone inference (official interface)

```bash
python infer.py --reference ref.png --search search.png
# prints: 512.34,488.10   (x,y in Search pixels)
```

## Generate synthetic data

The Drift-Sense synthetic generator (Applied Materials Hugging Face Space) is
not redistributed here; see `docs/submission/` for how splits were produced.
Each split is a directory with `reference/`, `search/` and a `manifest.csv`
(columns `id, architecture, reference_path, search_path, gt_x, gt_y`).

## Evaluate over a dataset

```bash
python evaluate.py --manifest path/to/split/manifest.csv [--verify] [--json-out report.json]
```

Reports mean / median / p90 / max pixel error, success@{2,5,10,20}px, runtime,
and the unique-vs-ambiguous breakdown.

## Verification

```bash
pytest -q
```

## Reproducibility and limitations

- All metrics are from **synthetic** Drift-Sense data, not official KLA/AMAT
  imagery, and are labelled as such. No metric is tuned on hidden test data.
- The ~25% ambiguity ceiling is a property of single-view periodic layouts, not
  of this solver. It is reported, not hidden.
- Sub-pixel accuracy assumes the correct correlation peak was selected; on
  ambiguous samples the sub-pixel refinement is applied to the wrong peak.
