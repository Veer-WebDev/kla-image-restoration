# Experiment Log

Only measured experiments appear here. None were run on the official KLA data, which is absent from this workspace.

| Experiment | Data / split | Seed | Model / loss | Measured result | Decision |
|---|---|---:|---|---|---|
| `smoke_e2e` | Local fixture images, validation | 0 | Residual U-Net, Charbonnier + 0.2 SSIM | PSNR 17.562, SSIM 0.888, MAE 0.1129. LPIPS unavailable. | Pipeline smoke evidence only. |
| `smoke_clean` | Local fixture images, source-level validation | 42 | 0.122 M parameter residual U-Net, one CPU epoch, Charbonnier + 0.2 SSIM | PSNR 18.570 vs bicubic 18.563, SSIM 0.89001 vs 0.88998, MAE 0.09963 vs 0.09967. 4.9 s training. | End-to-end CLI validation only. |
| `baseline_residual_unet` | 80 repository-generated synthetic wafer pairs, frozen validation split | 42 | 7.762 M parameter residual U-Net, 40 epochs, AdamW 2e-4, cosine, AMP | Model PSNR 25.703 vs bicubic 25.006, SSIM 0.897 vs 0.713, LPIPS 0.149 vs 0.389, MAE 0.0313 vs 0.0423. Tesla T4, 491.1 s training, 1.8 GiB peak. | Validates baseline engineering only. |
| `baseline_residual_unet__eval_test` | Same synthetic source, separate 8-image frozen test split | 42 | Best checkpoint selected by validation | PSNR 20.273 vs bicubic 19.978, SSIM 0.793 vs 0.688, LPIPS 0.226 vs 0.402, MAE 0.0845 vs 0.0897. Latency 18.03 ms/image mean on Tesla T4. | Honest proxy only. PSNR win rate was 37.5%. |

Raw ledger: `results/experiments.csv`. Detailed Colab artefacts: `results/colab_t4_40ep/`.

## Required next experiments once official pairs are available

1. Establish an official paired-data bicubic baseline.
2. Run A0–A4 augmentation ablation while holding split, seed, model and loss fixed.
3. Run Charbonnier versus Charbonnier+SSIM loss ablation.
4. Run the six-order degradation matrix and severity robustness matrix.
5. Save the selected checkpoint, per-image metrics, full-resolution maps and end-to-end runtime.

Do not transpose synthetic results into official results, and do not use hidden-test inputs for selection.
