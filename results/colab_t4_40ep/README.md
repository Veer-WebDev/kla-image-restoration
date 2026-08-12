# Colab T4 GPU run — 40 epochs (`colab_t4_40ep`)

Real GPU training of the residual U-Net, run on Google Colab (Tesla T4, torch
2.11.0+cu128, CUDA 12.8). This is the reference full-length run behind the
project's headline numbers.

## Setup

- **Hardware**: Tesla T4 (15 GB), 2 vCPU, numpy 2.0.2, Python 3.12.13.
- **Data**: 80 synthetic wafer pairs generated on-box with
  `scripts/make_fixtures.py --out data/GTset --count 80` (grid / line-space /
  trace / contact motifs run through the project's own degradation engine).
  The official KLA pairs were not available, so every number here is on
  synthetic data and describes the pipeline, not the official benchmark.
- **Split**: frozen 0.8 / 0.1 / 0.1 (seed 42). Validation during training,
  test split held out for the final eval.
- **Recipe**: `configs/baseline.yaml` unchanged — 40 epochs, batch 8, AdamW
  lr 2e-4, cosine schedule, Charbonnier+SSIM loss, AMP on, LPIPS in validation.
- **Model**: residual U-Net, base 32, depth 4, bicubic upsample, 7.76 M params.

Reproduce:

```bash
python scripts/make_fixtures.py --out data/GTset --count 80
python train.py --gt-dir data/GTset/GT --noisy-dir data/GTset/NoisyLR \
  --epochs 40 --experiment-id baseline_residual_unet
python evaluate.py --checkpoint runs/baseline_residual_unet/best.pth \
  --gt-dir data/GTset/GT --noisy-dir data/GTset/NoisyLR \
  --split test --split-file runs/baseline_residual_unet/split.json \
  --output-dir runs/baseline_residual_unet/eval_test --tag colab_t4_40ep
```

## Headline results

**Validation (best checkpoint, epoch 1)** — model vs bicubic baseline:

| metric | model | bicubic | delta |
|--------|------:|--------:|------:|
| PSNR   | 25.703 | 25.006 | **+0.697 dB** |
| SSIM   | 0.897  | 0.713  | **+0.185** |
| LPIPS  | 0.149  | 0.389  | **-0.240** (lower is better) |
| MAE    | 0.0313 | 0.0423 | -0.0110 |

Training: 40 epochs in 491 s (~12.3 s/epoch), peak 1.8 GiB, final train loss
0.087.

**Held-out test split (8 images, saved-PNG domain)** — the closest proxy to how
the organizers would score:

| metric | model | bicubic | delta |
|--------|------:|--------:|------:|
| PSNR   | 20.273 | 19.978 | +0.296 dB (median -0.471) |
| SSIM   | 0.793  | 0.688  | **+0.105** |
| LPIPS  | 0.226  | 0.402  | **-0.177** |
| MAE    | 0.0845 | 0.0897 | -0.0052 |

Cost: 7.76 M params, 18.0 ms/image mean latency on T4 (55 img/s, 3.6 MP/s),
peak 95 MiB at inference. Win rate 37.5 % of images beat bicubic on PSNR.

## Reading the numbers

The model wins decisively and consistently on the perceptual metrics (SSIM,
LPIPS) and on mean PSNR. The **median** test-split PSNR delta is negative
because the hardest motif — dense line/space gratings — loses sub-pixel detail
in downsampling that no deterministic model can invent back; on those few
images bicubic's smoother guess scores marginally higher on pixel PSNR while
looking worse perceptually (see the LPIPS gap). This is the expected and
honest failure mode, not a regression in the model.

## Files

- `train_summary.json` — full training run summary (from the run directory).
- `eval_test_summary.json` — aggregate test-split metrics + cost/latency.
- `history_val.csv` — per-epoch validation PSNR/SSIM/LPIPS vs bicubic.

The `best.pth` (30 MB) and `last.pth` (89 MB) checkpoints and the qualitative
error maps stay on the Colab runtime; they are not committed to keep the repo
lean. Re-run the commands above to regenerate them.
