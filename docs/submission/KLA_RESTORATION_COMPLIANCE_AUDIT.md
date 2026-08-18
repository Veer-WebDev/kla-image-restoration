# KLA Restoration Submission Compliance Audit

**Audit date:** 2026-08-17  
**Authoritative source reviewed:** `KLA_Problem Statement_Studen help document.pdf`  
**Scope reviewed:** repository `master` at commit `5f1526d`

## Verdict: do not submit this revision to the KLA restoration track

The supplied KLA help document defines an **AI image-restoration** task. The
current repository is a **Drift-Sense reference-localization** solution. It
accepts a Reference image plus a Search image and emits one coordinate. That is
a different input contract, output contract, evaluation metric, and deliverable
set. Passing localization tests or producing a localization architecture PDF
would not satisfy the restoration submission requirements.

This is a scope conflict, not a small documentation issue. This repository must
not be packaged or represented as a valid KLA restoration submission until the
blocking items below are resolved.

## Requirement-by-requirement audit

| KLA restoration requirement | Current repository state | Result |
|---|---|---|
| Restore each NoisyLR image to GT resolution | `infer.py` takes `--reference` and `--search`, then prints `(x,y)` | **Blocker** |
| Inference accepts `--input_dir` and `--output_dir`, saves each restored image | No current `inference.py`; `infer.py` writes no restored image | **Blocker** |
| GPU-capable image-restoration model | Default is CPU OpenCV NCC, explicitly no GPU/DL | **Blocker** |
| Training script reproduces submitted checkpoint | No current `train.py`; current package contains no trainable restoration model | **Blocker** |
| Final weights and model configuration included | No final restoration checkpoint/configuration tracked | **Blocker** |
| Treat benchmark degradation requirements as only speckle, Gaussian noise, downsampling | Current localization generator adds unrelated SEM controls: charging, barrel distortion, rotation, periodic layout and scale variation | **Blocker** |
| Paired GT/NoisyLR training workflow and leakage-free validation | Current data format is Reference/Search localization pairs, not paired GT/NoisyLR restoration data | **Blocker** |
| PSNR, SSIM and LPIPS reporting, with baseline comparison | Current results report coordinate error and localization success rate | **Blocker** |
| Full-resolution restored examples and failure cases | Current examples/results are localization coordinates and correlation ambiguity | **Blocker** |
| End-to-end restoration runtime, batch size, hardware and timing method | Current timings are CPU NCC localization per pair | **Blocker** |
| Solution PPT/PPTX covering restoration approach, losses, metrics, runtime and outputs | Current `model_architecture.pdf` describes localization; no compliant restoration PPT/PPTX is present | **Blocker** |
| README/environment commands for restoration interface | README describes Drift-Sense localization | **Blocker** |

## Existing recoverable material

Git history contains a previous restoration implementation before commit
`555a336` reshaped the repository for localization. The last richer restoration
commit identified in history is `c668591` (`Add model registry (edsr, nafnet) +
per-arch configs; fix silent YAML config drop`). It includes `train.py`,
`inference.py`, `src/kla_restore/`, restoration configuration and dependency
specifications.

This history is a **starting point only**, not evidence of a compliant final
submission. The old README explicitly says that official paired KLA training
images and a final competition checkpoint were not present. Git history does
not contain a final tracked `.pth`/`.pt` model weight file. Local ignored smoke
and CPU-comparison weights exist under `runs/`, but they must not be presented
as final KLA-trained weights unless their exact data, configuration, validation
split and results are rebuilt and documented.

## Required recovery path before submission

1. Preserve the current Drift-Sense localization work as an archive branch/tag.
2. Restore or reimplement the restoration package on a dedicated restoration
   submission branch.
3. Use the official paired GT/NoisyLR training data. Split by source/image before
   synthetic augmentation. Never train on hidden NoisyLR test inputs.
4. Limit benchmark claims and core training degradation to additive Gaussian
   noise, speckle noise and downsampling in mixed orders. Treat any other effect
   as a separately labelled, non-benchmark ablation only.
5. Train from a fixed configuration and seed. Retain the chosen final checkpoint,
   config, source-level split manifest, experiment ledger and checkpoint SHA-256.
6. Implement and exercise the required `inference.py --input_dir ... --output_dir
   ...` interface in a clean environment with the final checkpoint available.
7. Report measured validation PSNR, SSIM, LPIPS, baseline comparison, end-to-end
   runtime, batch size, GPU model, software versions and timing method. Do not
   reuse the current localization metrics.
8. Create the required restoration PPT/PPTX with full-size restored images,
   honest failures, external-resource disclosures and the repository link.
9. Check portal naming, upload-size limits and deadline separately. The supplied
   document states a Phase 1 date of 16 August 2026, but the portal is the
   authoritative source for exact cutoff and any extension.

## What can be truthfully claimed now

- A localization research prototype exists and is reproducible for its own
  Reference/Search coordinate task.
- A historical restoration codebase can be recovered from Git.
- No current claim of KLA restoration eligibility, official-score performance,
  final checkpoint availability or restored-image quality is justified.

## Submission gate

Do not create an upload archive or present `master` as a KLA restoration
submission until every blocker above is replaced by direct, reproducible
artifact and execution evidence.
