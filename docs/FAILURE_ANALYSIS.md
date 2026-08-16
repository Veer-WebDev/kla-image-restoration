# Failure Analysis

## Observed failure in the committed synthetic evaluation

The `baseline_residual_unet__eval_test` experiment had an 8-image held-out synthetic test split. The model's mean PSNR improved over bicubic by 0.296 dB, but its median PSNR delta was **-0.471 dB** and it won PSNR on only 37.5% of images. The run summary attributes the hard cases to dense line/space grating motifs.

The saved evaluation harness produces the required diagnostic set for every evaluated sample:

- NoisyLR input
- bicubic baseline
- model prediction
- GT
- absolute error map
- residual map, `prediction - bicubic`
- improvement map, `|bicubic - GT| - |prediction - GT|`

Representative generated maps from the local smoke evaluation live under `runs/_eval_smoke/maps/`. They are fixture diagnostics, not official examples.

## Interpretation

Severe downsampling removes high-frequency phase and line-spacing information. Multiple high-resolution structures can map to the same low-resolution observation, so no deterministic method can know which original was correct from the degraded pixels alone. A restoration model that invents a sharp grating may look plausible but cannot be claimed accurate without ground truth. On these cases, bicubic's smoother estimate can score slightly better on pixel PSNR even when the residual network improves SSIM or LPIPS on average.

## Required official-data follow-up

After official validation data is available, select the worst sample by declared metric, save its full-resolution diagnostics to `results/hard_cases/`, and report the observed cause without extrapolating beyond the evidence.
