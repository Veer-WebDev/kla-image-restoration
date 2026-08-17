# KLA Restoration Recovery Design

**Date:** 2026-08-17  
**Status:** approved recovery direction, pending final design review  
**Goal:** produce the smallest honest, reproducible KLA image-restoration package that meets the supplied help document's executable and documentation contract by the stated deadline.

## Scope boundary

The supplied Hugging Face Space is a Drift-Sense localization generator, not a
native KLA restoration dataset. It will therefore be used only as a disclosed
source-image collection: its high-resolution Reference images may become clean
source images (GT) for a *new*, deterministic restoration-pair generator.

The generated NoisyLR inputs will apply only the benchmark's three mechanisms:

1. additive Gaussian noise;
2. multiplicative speckle noise; and
3. downsampling.

Their ordering will be sampled from all six permutations. No location labels,
Search images, charging, barrel distortion, rotation, RGB bonus colors, or
localization metrics will be used in restoration training or reported as KLA
restoration evidence. No generated sample will be labelled as official KLA
training data.

## Architecture

Use the recoverable residual U-Net implementation from Git commit `c668591` as
the baseline final candidate. It is small enough for quick GPU training and has
a compliant restoration direction:

```text
NoisyLR -> bicubic upsample -> residual U-Net -> residual prediction
        -> bicubic + residual -> clipped restored image
```

The immediate target is one stable model, not an ensemble or an unvalidated
architecture search. Train against GT using Charbonnier plus SSIM loss only if
the restored code's existing implementation is verified. A bicubic-only output
is the required baseline. Gradient/frequency losses, model conditioning and
external pretrained weights are out of this time-box unless separately measured
and documented.

## Data and split design

1. Materialize source images from the existing local clone of
   `aayushraina21/drift-sense-synthetic-data`, or obtain the Space's public
   files without copying its source code.
2. Hash each source image and split by source identity, never by crop/view.
   Fixed seed determines train/validation/test source sets.
3. Create one or more GT crops per source only within the assigned split.
4. For every GT crop, create deterministic NoisyLR data using the three allowed
   degradations, a recorded seed, parameters and order.
5. Emit CSV/JSON manifests including source hash, split, crop, seed, Gaussian
   sigma, speckle sigma, scale and operation order.
6. Disclose the source URL, access date, observed licence status and the fact
   that source imagery is not official KLA data. If licence or availability is
   unsuitable, use only repository-generated clean motifs and label that fact.

## Interfaces and artifacts

The restoration branch will expose:

- `train.py --gt-dir ... --noisy-dir ...` plus a deterministic generated-pair
  training route;
- `inference.py --input_dir ... --output_dir ... --checkpoint ...`, with no
  manual source edit or local data path required;
- `weights/final_model.pth`, its SHA-256, model configuration and inference
  scale/output rules;
- `requirements.txt`, source-level validation manifest and experiment ledger;
- a result report with baseline/final PSNR, SSIM, LPIPS when available, MAE,
  runtime methodology, batch size, device and package versions;
- restored-image examples at native output resolution, including at least one
  failure; and
- a 12-slide `solution_presentation.pptx` following the supplied KLA outline.

No result will be called official KLA performance. If LPIPS pretrained weights
cannot be installed or downloaded in the available environment, that limitation
will be stated and the package will not falsely claim LPIPS was measured.

## Migration and safety

1. Tag the current localization state before changing the active submission
   surface.
2. Create a dedicated restoration branch.
3. Restore the pre-localization restoration files from `c668591`.
4. Remove or move localization-facing entry points from the restoration branch
   so evaluator-facing README and script names are unambiguous. Retain the
   archive tag untouched.
5. Add a reproducible pair generator and data card. Then run tests before
   training.
6. Train and select only on the validation split. Evaluate the selected
   checkpoint once on a held-out synthetic test split.
7. Exercise full clean-environment inference with only input/output arguments
   plus the bundled checkpoint. Package only after artifact checks pass.

## Failure handling

- If the Hugging Face assets cannot be legally used or their source images are
  unavailable, stop using them and rely on a disclosed in-repo clean-motif
  generator. Do not reproduce organiser code.
- If a final training run cannot complete before the deadline, retain the model
  and its actual partial-run status rather than fabricate metrics or training
  provenance.
- If the official dataset is obtained later, retrain from scratch, retain an
  untouched holdout, and replace the provisional synthetic-data results.

## Acceptance checks

A package may be described as an *honest provisional KLA restoration
submission package* only after all of these run successfully:

1. clean-environment dependency install;
2. `train.py` or documented reproduction command produces the final checkpoint;
3. `inference.py --input_dir --output_dir --checkpoint` writes one valid output
   per input with preserved stems and documented dimensions/range;
4. baseline and final metrics are computed on a source-disjoint held-out split;
5. checkpoint/config/hash/manifest/results/PPTX exist and agree; and
6. README commands and external-resource disclosures match the actual package.

It is not valid to claim competitive hidden-test performance or official KLA
training-data usage without those unavailable assets.
