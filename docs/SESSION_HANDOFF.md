# Session Handoff — KLA Restoration Submission

_Last updated: 2026-08-18 (UTC). Branch: `kla-restoration-submission`._

This document captures the complete working state so another agent (or a
future session) can resume without re-deriving context. It complements the
plan and audit docs already in `docs/`.

## Goal

Finalize the KLA (SEMICON India 2026) AI-based semiconductor image restoration
submission against the organizer's `run.py` + `.npy` evaluator contract, and
train a stronger `models/best.pth` checkpoint on Google Colab (T4 GPU).

## Evaluator contract (implemented, committed)

`run.py` is the evaluator entry point:

- Invocation: `python run.py <input-dir> <output-dir>` (positional args).
- Reads every `.npy` under the input dir recursively.
- Writes one `.npy` per input to the output dir, preserving the relative
  filename.
- Output is `(H, W)` float32 in `[0, 1]`, NaN/Inf scrubbed, target size =
  input size x checkpoint scale (default 2).
- Fully offline, no downloads, no API keys, GPU auto-detected (CPU fallback).
- Loads bundled weights from `models/best.pth` via
  `src/kla_restore/checkpoint.py:load_model`; device via
  `src/kla_restore/utils.py:select_device`.

Validated end-to-end in the project venv against 2D, `(H,W,1)`, 3-channel,
and NaN/Inf inputs — all pass filename parity, shape, dtype, `[0,1]` range,
and finiteness.

## Current checkpoint status (IMPORTANT)

`models/best.pth` bundled in the repo is a **placeholder**: 7,477 params
(base_channels=4), epoch 1, at roughly bicubic parity. It is a valid contract
artifact but weak. The Colab run replaces it with the real base_channels=48
model (~17.5M params) selected on validation PSNR.

## Training plan (ready to run on Colab)

- Config: `configs/submission_big.yaml`
  - Corpus: 160 first-party synthetic clean sources -> 768 train / 96 val /
    96 test paired views, source-disjoint by SHA-256 before view expansion.
  - Model: residual U-Net, base_channels=48, depth 4, bicubic upsample.
  - Train: 60 epochs, batch 16, AdamW lr 3e-4, cosine schedule, 2 warmup
    epochs, grad clip 1.0, AMP, early-stop patience 12, selection metric PSNR.
  - Loss: Charbonnier + 0.2 * SSIM.
- Notebook: `notebooks/colab_train.ipynb` (updated). Cells, in order:
  1-2. Record runtime + `nvidia-smi`.
  3-4. Upload `kla_restoration_submission.zip`, unzip, `cd` in.
  5-6. `pip install -r requirements.txt`.
  7-8. Build corpus: `generate_clean_sem_sources.py` (count 160, size 768) +
       `materialize_restoration_data.py` (views 6, crop 512, scale 2).
  9-10. Train: `python train.py --config configs/submission_big.yaml`.
  11-12. Copy validation-best to `models/best.pth`, sha256, evaluate ONCE on
         held-out test -> `results/submission_big/summary.json`.
  13-14. Smoke-test the `run.py` `.npy` contract on the test NoisyLR dir.
  15-16. Zip + download `kla_restoration_big_artifacts.zip`.

### To run

1. Upload `notebooks/colab_train.ipynb` to Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Run cells top to bottom. Cell 4 prompts for `kla_restoration_submission.zip`
   (repo-root archive of HEAD).
4. Training ~20-40 min on T4.
5. Cell 16 downloads the artifact zip.

### Report back after the run

- Cell 10 final val PSNR (the `new best` line).
- Cell 12 `results/submission_big/summary.json` (test PSNR/SSIM vs bicubic).
- The artifact zip, so the trained `models/best.pth` can be dropped into the
  repo and the `run.py` contract re-verified before finalizing.

## Environment notes

- Local Windows system Python is 3.14 without torch. Use the project venv
  `.venv\Scripts\python.exe` (Python 3.11, torch 2.5.1+cpu) for torch work.
- There is **no local GPU**, and local CPU training crashed with an access
  violation (exit code -1073741819). Colab is the only viable training path.
- `requirements.txt` pins torch==2.5.1, numpy, Pillow, PyYAML, tqdm, plus eval
  extras. CUDA wheels are noted for Colab.

## Honesty guardrail

All PSNR/SSIM numbers to date are **synthetic-pipeline evidence only**, not
official KLA scores and not hidden-test performance. `data/*` synthetic corpora
carry this warning in their dataset cards. Do not present synthetic results as
official.

## Reproducible upload archive

`kla_restoration_submission.zip` is committed at repo root: a `git archive` of
the finalized HEAD (78 files) containing `run.py`, `models/best.pth`,
`requirements.txt`, `configs/submission_big.yaml`, `notebooks/colab_train.ipynb`,
and the data generation scripts. Rebuild with:

```
git archive --format=zip -o kla_restoration_submission.zip HEAD
```

## Key files

| Path | Purpose |
| --- | --- |
| `run.py` | Evaluator `.npy` entry point (positional args). |
| `models/best.pth` | Bundled checkpoint (placeholder until Colab run). |
| `configs/submission_big.yaml` | Bigger-corpus training config. |
| `notebooks/colab_train.ipynb` | End-to-end Colab training + verification. |
| `train.py` / `evaluate.py` / `inference.py` | Training/eval/inference CLIs. |
| `scripts/generate_clean_sem_sources.py` | Clean synthetic source generator. |
| `scripts/materialize_restoration_data.py` | Paired GT/NoisyLR materializer. |
| `src/kla_restore/` | Model, data, train, checkpoint, utils. |
| `docs/` | Data card, audit, verification, experiment log. |
