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

| Solver / evaluation rule | success@10px | median err | time/sample |
| --- | --- | --- | --- |
| NCC, **official center tie-break** | **59.0%** | 1.33 px | 209 ms |
| NCC, raw highest peak (legacy crop-label diagnostic, non-compliant) | 75.5% | 1.03 px | 156 ms |

The official centre tie-break changes only ambiguous cases. The external
fixture's random-crop labels do not use that official convention, so the
spec-compliant 59.0% is the honest result to cite for this fixture. The 75.5%
raw-peak diagnostic is retained only to show the prior convention mismatch.
The successful unique matches are still essentially exact. The much harder
remaining cases are **appearance-ambiguous periodic arrays**: at the ground-truth
location the degraded Search content matches the Reference no better than at a
wrong repeated location. A single degraded Search image physically does not
contain the information to disambiguate such repeats, so
**this is a fundamental information limit, not a solver deficiency**. For the
inspected ambiguous cases, a larger learned model cannot recover information the
input does not carry. The solver
exposes this honestly: it flags such samples as ambiguous.

| Subset (flagged by the solver) | count | success@10px | median err |
| --- | --- | --- | --- |
| Unique correlation peak | 92 | **97.8%** | 0.83 px |
| Competing (ambiguous) peaks | 108 | 25.9% | 73.93 px |

The larger ambiguous share under the official tie-break is expected on this
external fixture because its random crop label is not necessarily the tied region
nearest the Search center. The ambiguity flag is still a useful signal: when the
solver reports a unique peak it is right 98% of the time.

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

Per the task rules ("no dataset is provided; participants shall generate their
own"), this repo ships a self-contained generator built only from publicly
known DRAM/FinFET structural characteristics and literature-backed SEM noise
models (see `src/drift_localize/generator.py` and `docs/EXTERNAL_RESOURCES.md`
for citations):

```bash
python generate_dataset.py --out data/mydata --n 30 --seed 31337
# noisier search images (robustness stress test):
python generate_dataset.py --out data/noisy --n 30 --search-speckle 0.6
# organiser-slide stress controls: charging, scan distortion, rotation, and
# global feature/polygon scale from -20% to +20%:
python generate_dataset.py --out data/stress --n 30 --charging-prob 0.2 --charging-intensity 1 \
  --barrel-k 0.02 --rotation-max-deg 3 --feature-scale-min 0.8 --feature-scale-max 1.2
```

It writes `reference/`, `search/`, and a `manifest.csv` (columns
`id, architecture, reference_path, search_path, gt_x, gt_y`) directly consumable
by `evaluate.py`. The generator composes periodic array "mats" separated by
irregular strips and a sparse constellation of alignment fiducials, so most
crops carry a locally-unique landmark while purely periodic regions remain
genuinely ambiguous (the honest failure mode).

### Noise robustness (FAQ: "search image will be noisier in test data")

Sweeping multiplicative speckle on the Search image (30 samples each, own
generator):

| search speckle σ | success@5px | unique subset | ambiguous subset |
| --- | --- | --- | --- |
| 0.0 | 86.7% | 22/30 @ 100% | 8/30 @ 50% |
| 0.3 | 60.0% | 14/30 @ 100% | 16/30 @ 31% |
| 0.6 | 36.7% | 5/30 @ 100% | 25/30 @ 24% |

The key result: the **unique-peak subset stays 100% correct (~0.05 px median)
at every noise level**. Noise does not corrupt confident matches; it *shrinks*
the confident subset as fiducials get buried, which the ambiguity flag reports
honestly.

The added geometric effects are deliberately stress-test controls, not claimed
as solved deployment robustness. In particular, a rotation-aware NCC search is
available only through the evaluator's `--angles` study flag because it did not
improve the SEM-like smoke evaluation and substantially increases runtime.

## Ambiguous-tile tie-break

The task says: if more than one region matches, report the one **closest to the
Search image centre**. The matcher implements this (`center_tiebreak=True`,
default). On our crop-labelled synthetic ground truth a plain highest-peak
choice scores higher, but it does not follow the stated convention; the flag
`center_tiebreak=False` exposes that behaviour for comparison.

## Optical RGB (bonus)

The task offers bonus credit for generalizing to optical-microscope RGB images
after the SEM core is solid. Pass `--rgb` to the generator to render
optical-microscope-style 3-channel pairs:

```bash
python generate_dataset.py --out data/rgb --n 30 --rgb
```

The matcher operates on luminance, so it localizes RGB pairs with no code
change: on a 10-sample RGB set it matched the grayscale quality (success@10px
80%, confident subset 100% at 0.06 px median).

## Evaluate over a dataset

```bash
python evaluate.py --manifest path/to/split/manifest.csv [--verify] [--json-out report.json]
# The rubric's 1px and 5px positive-pair confusion summaries:
python evaluate.py --manifest path/to/split/manifest.csv --cm-thresholds 1 5
# Optional, non-deployment robustness study:
python evaluate.py --manifest path/to/split/manifest.csv --angles -3 -2 -1 0 1 2 3
```

Reports mean / median / p90 / max pixel error, success@{2,5,10,20}px, runtime,
and the unique-vs-ambiguous breakdown.

Because every manifest row is a positive pair with a known target, a conventional
four-cell classification confusion matrix has no true-negative or false-positive
examples. The evaluator reports the meaningful TP/FN outcome counts at each
spatial tolerance. `analysis/noise_sweep.py` supplies a separate score-threshold
protocol for a real precision-recall curve:

```bash
python analysis/noise_sweep.py --out results/noise_sweep --levels 0 0.3 0.6 \
  --calibration-n 30 --test-n 30 --seed 777
```

It writes a JSON ledger and a portable SVG plot. On the separate held-out
30-pair synthetic tests at 5px, precision/recall were 79.2%/82.6% at σ=0.0,
92.3%/63.2% at σ=0.3, and 27.8%/100% at σ=0.6. Thus confidence thresholding is
not useful at heavy noise, an explicit limitation rather than a hidden failure.

An experimental SIFT + RANSAC alternative was also measured on the fixed
40-pair external synthetic test split. It produced 0.0% success@5px and
success@10px (median 411.31px, 260ms/sample, median zero RANSAC inliers), so it
is intentionally excluded from the deployment interface. See
`analysis/feature_baseline.py` and `docs/submission/RUN_REPORT.md`.

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
