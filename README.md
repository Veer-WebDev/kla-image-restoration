# KLA Semiconductor Image Restoration

A deterministic, submission-oriented baseline for the **KLA SEMICON India Hackathon 2026** restoration task.

> **Status on 2026-08-17:** the repository contains a recovered and regression-tested residual U-Net baseline, a first-party source-disjoint synthetic corpus generator, and historical synthetic Colab evidence. Official paired KLA images are not present. Therefore, no repository result is represented as an official benchmark score or hidden-test performance.

## Method

```text
NoisyLR → bicubic resize to target size → residual U-Net → bicubic + residual → clamp [0, 1]
```

The model uses convolution, GroupNorm, GELU and skip connections. It predicts a correction rather than a whole image, which gives a stable bicubic fallback. The forward degradation engine implements only the three specified mechanisms: additive Gaussian noise, multiplicative speckle noise and downsampling. It covers all six operation orders deterministically and intentionally does **not** clip NoisyLR values.

## Repository layout

- `train.py`: deterministic training, resume and experiment ledger entry point.
- `inference.py`: evaluator-facing offline inference CLI. It imports no LPIPS or scikit-image.
- `evaluate.py`: paired evaluation, bicubic baseline, runtime, maps and hard-case selection.
- `configs/baseline.yaml`: controlled starting recipe. Unspecified degradation priors live in `configs/degradation.yaml`.
- `src/kla_restore/`: model, data, degradation, metrics, checkpoints and training logic.
- `tests/`: pairing, splitting, normalization, deterministic degradation, metrics, shape and seed regressions.
- `docs/`: audit, external-data gate, experiment record, augmentation rationale, limitations and requirement-to-evidence traceability.

## Installation

Python 3.10+ is required. Create an environment and install the package:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e '.[eval,dev]'
```

For a CUDA build, install the PyTorch wheel matching the target CUDA runtime **before** installing the remaining requirements. The inference code falls back to CPU when CUDA is unavailable.

## Data contract

Pass the official paired directories directly. Files are found recursively and paired by a canonical filename stem. Known role suffixes, for example `_gt`, `_clean`, `_noisylr` and `_lr`, are removed only for pairing. Duplicate or missing stems are reported and cannot silently become evaluation samples.

```text
/path/to/KLA/
├── GT/
│   └── sample_001_gt.png
└── NoisyLR/
    └── sample_001_noisylr.png
```

GT is dtype-normalized and clipped to `[0, 1]`. NoisyLR has the same dtype-normalization but is deliberately left unclipped. Images may be grayscale or RGB. Training/validation/test splitting happens by source stem before synthetic views are generated.

## First-party synthetic corpus

When official paired training data is unavailable, create only the disclosed first-party corpus below. The clean-source generator contains no third-party images. The materializer source-splits by clean-image SHA-256 before it produces any crop or degraded view, applies only Gaussian, speckle and downsampling, cycles all six operation orders and retains unclipped NoisyLR arrays as `float32 .npy`.

```bash
python scripts/generate_clean_sem_sources.py \
  --out data/restoration_sources --count 96 --size 768 --seed 20260817
python scripts/materialize_restoration_data.py \
  --source-dir data/restoration_sources --out data/kla_restoration \
  --seed 20260817 --views-per-source 6 --crop-size 512 --scale 2
```

Read `docs/DATA_CARD.md` before interpreting any metric. This corpus validates the public pipeline only. It is not official KLA data.

## Train

First inspect the pairing report and a small overfit run. Then run the fixed baseline recipe:

```bash
python train.py \
  --gt-dir /path/to/KLA/GT \
  --noisy-dir /path/to/KLA/NoisyLR \
  --config configs/baseline.yaml \
  --experiment-id official_baseline_seed42
```

Training stores the resolved configuration, source-level split, validation manifest, CSV history, last checkpoint and best-validation checkpoint under `runs/<experiment_id>/`. Every completed run appends an actual result row to `results/experiments.csv`.

Do not tune against the hidden test data. Run ablations one variable at a time and keep only measured results.

## Evaluate

```bash
python evaluate.py \
  --checkpoint runs/official_baseline_seed42/best.pth \
  --gt-dir /path/to/KLA/GT \
  --noisy-dir /path/to/KLA/NoisyLR \
  --split val \
  --split-file runs/official_baseline_seed42/split.json \
  --output-dir runs/official_baseline_seed42/eval_val
```

This records PSNR, SSIM, LPIPS when its pretrained weights are available, MAE, a bicubic comparison, end-to-end latency, per-image metrics, residual maps, absolute-error maps and improvement maps. LPIPS is evaluation-only and never required by `inference.py`.

## Standalone inference

```bash
python inference.py \
  --input_dir /path/to/KLA/test/NoisyLR \
  --output_dir /path/to/submission \
  --checkpoint runs/official_baseline_seed42/best.pth \
  --scale 2
```

The two directory arguments are mandatory. By default, every output preserves its input-relative filename and extension, and pre-existing outputs fail before model execution unless `--overwrite` is explicit. Target size is resolved in this strict order:

1. `--target-size H W`
2. `--size-map file.csv|json`
3. `--scale N`
4. the scale embedded in the checkpoint
5. scale `2` fallback

The correct test-time scale or size map must be confirmed from KLA's evaluator instructions before submitting. GT is unavailable at hidden-test inference time, so the script never tries to infer output dimensions from GT.

## Verification

```bash
pytest -q
# Linux/macOS clean-environment smoke test
bash scripts/smoke_test.sh
```

The smoke test creates an isolated environment, generates small deterministic fixtures, trains a tiny checkpoint, invokes the standalone inference CLI, checks output count/format/dimensions/range and exits nonzero on failure.

## Measured evidence currently committed

`results/colab_t4_40ep/` contains a real 40-epoch Tesla T4 run on 80 repository-generated **synthetic** wafer motifs, not on the official KLA data. Held-out synthetic test metrics were PSNR 20.273, SSIM 0.793, LPIPS 0.226 and MAE 0.0845, versus bicubic PSNR 19.978, SSIM 0.688, LPIPS 0.402 and MAE 0.0897. Mean end-to-end latency was 18.0 ms/image on that T4. See the result README and `docs/EXPERIMENT_LOG.md` for scope and limitations.

## External data and models

The supplied Hugging Face Space is an Applied Materials Drift-Sense synthetic-data generator, not a KLA restoration dataset. Its repository currently exposes no explicit licence file or competition-use grant. It is **not downloaded, trained on, or included** in this submission baseline. See `docs/EXTERNAL_RESOURCES.md`.

Any Kaggle or other external dataset must be entered in that document before use with its name, URL, licence, data card, intended role and an experiment ID. Only sources whose licences permit competition use may be used.

## Reproducibility notes and known limitations

- CUDA deterministic kernels are configurable with `strict_determinism`; exact cross-hardware bitwise equality is not guaranteed by PyTorch.
- Gaussian/speckle levels, downsample factors and kernels are initial priors until calibrated from official pairs. They are isolated in YAML.
- The `--scale 2` default is an explicit assumption, not an assertion about the hidden test set.
- Dense structures destroyed by downsampling cannot be deterministically recovered. Error maps must be reported rather than presenting hallucinated detail as fact.
- A final checkpoint trained only on synthetic sources remains evidence of a reproducible pipeline, not a credible claim of official KLA hidden-test performance. Retrain or validate only on data that the organizers permit before making a performance claim.
