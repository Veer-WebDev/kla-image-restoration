# Augmentation Justification

The KLA document confirms only additive Gaussian noise, multiplicative speckle noise and downsampling. It does not publish their level distributions, downsampling kernel or operation order. The repository therefore keeps all uncertain values in `configs/degradation.yaml` instead of scattering assumptions in code.

## Current controlled prior

- Gaussian sigma: 0.005–0.08 in normalized units.
- Speckle: `x * (1 + Normal(0, sigma))`, sigma 0.01–0.15.
- Downsampling scale: 2 or 4, with area, bicubic and bilinear kernels.
- Operation order: all six permutations.
- NoisyLR clipping: disabled.

These are starting priors inherited from the audited starter notebook, **not fitted facts about KLA data**. The only geometric transforms are flips and right-angle rotations, isolated to training and derived deterministically from the sample seed. No blur, JPEG artefacts, motion, colour jitter or arbitrary image augmentation is added.

## Calibration procedure after receiving official pairs

1. Quantify GT/NoisyLR scale, dimensions, channels and unbounded NoisyLR fraction.
2. Fit candidate noise and resampling priors using the training partition only.
3. Freeze those priors in a versioned config and record the source of every range.
4. Run A0 through A4 one change at a time on a fixed source-level validation split.
5. Retain an augmentation only when its validation improvement is measured and robust across relevant severities.

This policy prioritizes interpretable coverage of the specified degradation space over augmentation volume.
