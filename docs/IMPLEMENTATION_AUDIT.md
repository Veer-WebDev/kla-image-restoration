# Implementation Audit — KLA Image Restoration Starter Notebook

**Audited artifact:** `kla_restoration_colab_starter.ipynb` (26 cells, Colab, torch + lpips)
**Authority:** `KLA_Problem Statement_Studen help document.pdf` (SEMICON India Hackathon 2026)
**Audit date:** 2026-08-12
**Auditor role:** senior ML/CV engineer preparing the submission-ready repository

This document is written before any production code is authored. It records what the notebook
already gets right, what is incomplete, what is risky, and every assumption that the official
document does not settle. Nothing here is speculative about results: no metric is quoted because
no experiment has been run yet.

---

## 1. What currently works

| # | Item | Evidence in notebook | Verdict |
|---|------|----------------------|---------|
| 1 | Core restoration topology | Cell 10/12/21: `NoisyLR → bicubic upsample → residual U-Net → clamp` | Sound. Residual-on-bicubic is the correct low-risk formulation: the network only has to learn a correction, so at worst it degrades gracefully toward the bicubic baseline. |
| 2 | Three official degradations only | Cell 8: `gaussian`, `speckle`, `downsample` | Matches the specification exactly. No unofficial mechanisms (blur, JPEG, motion) are introduced. Correct scoping. |
| 3 | Randomized degradation order | Cell 8: `rng.shuffle(ops)` | Matches "their order is not disclosed". Covers all 6 permutations in expectation. |
| 4 | Seeded degradation | Cell 8: `np.random.default_rng(seed)` | Same seed reproduces the same degraded sample. This is the right primitive. |
| 5 | NoisyLR is not clipped | Cell 6: `load_image_float(..., clip=False)` | Matches "NoisyLR values may extend slightly outside [0,1]; this is intentional". Correct and easy to get wrong. |
| 6 | GT is clipped to [0,1] | Cell 6: `load_image_float(..., True)` | Matches "GT values are normalized to [0,1]". |
| 7 | Source-level split before augmentation | Cell 10: split on `keys` (image stems), then datasets are built per split | Correct in principle: no augmented copy of a training GT can appear in validation, because the split is on stems, not on generated samples. |
| 8 | Metric set | Cell 14: PSNR, SSIM, LPIPS, MAE | Matches mandated reporting plus the extra selection metric the document asks for. |
| 9 | Mandatory overfit sanity check | Cell 16: 300 steps on 2 samples | Exactly the pipeline sanity check the official student workflow (step 15) requires. |
| 10 | Charbonnier + SSIM loss | Cell 14: `charbonnier + 0.2*(1-ssim)` | Reasonable starting loss balancing pixel fidelity and structure. Must still be ablated. |
| 11 | AdamW + cosine + AMP + grad clip + best-checkpoint | Cell 18 | The right training scaffold. |
| 12 | Reproducibility metadata dump | Cell 23 | Captures python/platform/torch/cuda/seed/params. Good instinct, incomplete content (see §2). |
| 13 | Bicubic baseline comparison exists | Cell 21: prints bicubic vs model PSNR/SSIM | The mandated baseline comparison is present, though only qualitatively for one image. |

**Summary:** the notebook is a *correct skeleton*. Architecture, degradation model, loss, split
philosophy and metric set are all defensible and aligned with the official document. It is not a
submission, and it is not reproducible in its current form.

---

## 2. What is incomplete

1. **No standalone inference script.** The document mandates
   `inference.py --input_dir ... --output_dir ...` with no source edits. The notebook only has
   `restore_array` bound to in-notebook globals (`model`, `DEVICE`).
2. **No output-resolution contract at inference time.** `restore_array(noisy, target_hw)` requires
   the GT shape as an argument. At test time **GT is not available** — KLA supplies only degraded
   inputs. The notebook therefore has no working rule for choosing the output size. This is the
   single most dangerous gap: a wrong output size makes every metric unscoreable.
3. **No degradation-scale inference.** Related to (2): the model must decide the upsample factor
   from the input alone.
4. **No experiment tracking.** `history` is an in-memory list; nothing is written to
   `results/experiments.csv`. Every run's configuration is lost on disconnect.
5. **No ablations.** Augmentation ablation (A0–A4), loss ablation (L1 / Charbonnier /
   Charbonnier+SSIM), robustness matrix and degradation-order sweep are all absent.
6. **No test suite.** Zero tests. Pair discovery, determinism, shape preservation and split
   isolation are all unverified.
7. **No full-image validation.** Validation uses `training=False`, which skips cropping — good —
   but `val_loader` batches size 1 with no guarantee the U-Net can consume arbitrary
   (non-multiple-of-16) sizes. See §3.4.
8. **No runtime measurement.** The document defines end-to-end runtime as disk read →
   preprocess → H2D → forward → D2H → postprocess → save. The notebook measures nothing.
9. **No failure analysis.** No worst-case selection, no error maps, no residual maps.
10. **No dependency pinning.** Cell 1 installs `lpips scikit-image pyyaml tqdm` unpinned and
    relies on Colab's preinstalled torch. Not reproducible on the evaluator's machine.
11. **No external-resource disclosure.** LPIPS ships pretrained AlexNet weights; its license and
    provenance must be disclosed. Not documented anywhere.
12. **No config files.** Every hyperparameter and noise range is hardcoded in cells.
13. **No resume capability.** A Colab disconnect at epoch 30/40 loses the run.

---

## 3. What is risky

### 3.1 Grayscale collapse is silent and unjustified
Cell 6 converts any 3-channel input to luminance with fixed BT.601 weights and returns a
single-channel array. The official document never states the dataset is grayscale. If the official
dataset is RGB, this pipeline **destroys two thirds of the signal** and saves single-channel PNGs,
which will very likely mismatch the expected output format.

**Risk level: critical.** Mitigation: make channel handling explicit and data-driven, support
1- and 3-channel end to end, and record the detected channel count in the config snapshot.

### 3.2 `load_image_float` normalization heuristic is fragile
```python
if np.issubdtype(original_dtype, np.integer): arr /= np.iinfo(original_dtype).max
elif arr.max() > 1.5: arr /= 255.0
```
Three problems:
- The luminance conversion happens **before** dtype inspection, so `arr.dtype` after
  `0.299*R+0.587*G+0.114*B` is already `float64`. The integer branch is therefore **dead code for
  every colour image**, and such images fall into the `arr.max() > 1.5` heuristic instead.
- For a `float32` NoisyLR image legitimately stored in [0,1] with a max of, say, 1.04 (allowed by
  the spec!), the heuristic does not fire — correct — but for a float image stored in [0,255] with
  max 200 it divides by 255 — also correct — while a float image whose true max is 1.6 would be
  silently divided by 255 and destroyed. The heuristic is a coin flip on adversarial inputs.
- 16-bit PNGs divide by 65535, which is right, but only when the image is single channel.

**Risk level: high.** Mitigation: dtype-driven normalization decided *before* any channel math,
with the scale factor logged per image and overridable in config.

### 3.3 Speckle model is a modelling assumption, not a specification
Cell 8 implements speckle as `x *= (1 + N(0, σ))`. This is the standard "multiplicative Gaussian"
speckle used by MATLAB's `imnoise(...,'speckle')`. The official document says only "speckle noise".
It does **not** say the multiplier is Gaussian rather than Rayleigh/Gamma, and it does not give σ.

**Risk level: medium.** This is the most likely source of train/test distribution mismatch.
Mitigation: keep the multiplicative-Gaussian form as the declared default (it is the canonical
reading), isolate it in `configs/degradation.yaml`, and — critically — *calibrate the ranges from
the official paired data* rather than trusting the notebook's invented `(0.005, 0.08)` and
`(0.01, 0.15)`.

### 3.4 U-Net dimension safety is only half-solved
The decoder resizes to the encoder skip's exact shape via
`F.interpolate(x, size=t.shape[-2:])`, which does prevent shape mismatch on concatenation. Good.
But `MaxPool2d(2)` on an odd dimension floors it, and four successive pools mean a 5-pixel-wide
feature map region can collapse. More importantly `nn.GroupNorm(8, cout)` requires
`cout % 8 == 0`; with `base=16` in the debug cell (`ResidualUNet(16)`) the first encoder is
`GroupNorm(8, 16)` — fine — but any `base < 8` or odd base silently breaks. And a 1-pixel
dimension after four pools produces degenerate statistics.

**Risk level: medium.** Mitigation: explicit pad-to-multiple-of-16 at the model boundary with
crop-back to the exact requested output size, plus shape tests at 128/256/512 and odd sizes.

### 3.5 Output clamping is inside the model
Cell 12 returns `torch.clamp(x + self.out(d1), 0, 1)`. Clamping inside `forward` **zeroes the
gradient** for every pixel driven outside [0,1] during training. Early in training, when the
residual is poorly scaled, this can stall large image regions.

**Risk level: medium.** Mitigation: return the unclamped estimate during training and clamp only
at inference/evaluation, or expose `clamp_output` as a flag. This must be measured, not assumed —
it is a candidate ablation.

### 3.6 `torch.cuda.amp.GradScaler` / `autocast` are deprecated
On torch ≥ 2.4 these emit `FutureWarning` and the modern spelling is `torch.amp.GradScaler("cuda")`
and `torch.amp.autocast("cuda")`. Cosmetic today, breaking later.

**Risk level: low.**

### 3.7 CUDA nondeterminism is not fully pinned
`seed_everything` sets `cudnn.deterministic=True` and `benchmark=False`, which is necessary but not
sufficient. `F.interpolate(mode='bicubic'|'bilinear')` backward and `adaptive`/`area` kernels can
still be nondeterministic, and AMP changes accumulation order. There is no
`torch.use_deterministic_algorithms(...)` call and no `CUBLAS_WORKSPACE_CONFIG`.

**Risk level: medium for the reproducibility axis** ("Training & compute hygiene" is an explicit
evaluation axis). Mitigation: opt-in strict determinism flag, seeded DataLoader workers with a
`worker_init_fn` and a seeded `generator`, and an honest statement in the README about what is
bit-exact and what is not.

### 3.8 DataLoader worker seeding is absent
`DataLoader(..., num_workers=2)` with `shuffle=True` and no `generator=` and no `worker_init_fn=`.
Each worker inherits a different torch seed derived from the base seed, so it happens to be
deterministic *for a fixed worker count*, but changing `num_workers` changes the data stream. Runs
are therefore not comparable across machines with different worker settings.

**Risk level: medium.** Mitigation: explicit `generator` + `worker_init_fn`.

### 3.9 Validation "official" mode silently degenerates
Cell 10: `val_ds = RestorationDataset(val_keys, ..., 'official' if noisy_map else 'synthetic', ...)`.
But inside `__getitem__` the synthetic branch also triggers when `key not in self.noisy_map`. So in
`official` mode a key missing its NoisyLR partner silently becomes a *synthetic* validation sample —
changing the validation distribution without warning. Model selection would then be driven by a
partly synthetic set that varies with data completeness.

**Risk level: high for model selection integrity.** Mitigation: explicit pairing report, hard
failure or explicit logging when a requested official pair is missing, and a frozen validation
manifest written to disk.

### 3.10 One sample per source image per epoch
`__len__` returns `len(self.keys)`, and `__getitem__` derives its RNG from `self.seed + idx*1009`,
which is **epoch-invariant**. Consequences:
- With a small official dataset, an epoch is tiny and the cosine schedule over 40 epochs sees very
  few optimizer steps.
- The crop position and the synthetic/official coin flip are **identical every epoch**. So the
  "augmentation" is a fixed dataset of size N, not a distribution. The model sees the same 1 crop
  of each image forever. This substantially weakens the entire augmentation story.

**Risk level: high — this is the most important functional defect.** Mitigation: a
`samples_per_image` multiplier and epoch-aware seeding for training, while keeping validation
strictly fixed.

### 3.11 Synthetic and official samples are mixed but not balanced or logged
`synthetic_prob=0.6` is a hardcoded magic number with no justification and no per-epoch accounting
of how many official versus synthetic samples were actually consumed.

**Risk level: low-medium.** Mitigation: config + per-epoch counters in the CSV log.

### 3.12 Bicubic upsampling target uses `gt.shape` at train time
Cell 10: `F.interpolate(t, size=gt.shape, ...)`. Training therefore always knows the exact target
size. Inference does not. This trains the model on a privileged signal it will not have at test
time — not label leakage in the metric sense, but a **contract mismatch** that (2) must resolve.

**Risk level: critical (same root cause as §2.2).**

### 3.13 Checkpoint portability
`torch.save({'model_state': ..., 'metrics': ..., 'epoch': ..., 'seed': ...})` stores no
architecture description. Loading requires the reader to *know* `base=32` and the exact class
definition. If `base` is ever changed, old checkpoints load into the wrong graph or raise.

**Risk level: medium.** Mitigation: embed the full model config, a format version, the normalization
policy and the channel count inside the checkpoint.

### 3.14 LPIPS at inference time
`lpips.LPIPS(net='alex')` downloads pretrained weights on first use. The document forbids requiring
network access or manual steps in the evaluated pipeline. LPIPS must be **evaluation-only** and must
never be imported by `inference.py`.

**Risk level: high if violated.** The notebook imports lpips globally in cell 2, which means any
script derived from it inherits the dependency.

### 3.15 `skimage.metrics.structural_similarity` argument order
Cell 14: `ssim(t, p, data_range=1.0)`. SSIM is symmetric in its two image arguments, so the order is
harmless here, but `data_range=1.0` is asserted while predictions are clamped to [0,1] and GT is
clipped to [0,1] — consistent. However for multichannel images `channel_axis` must be supplied or
skimage raises. Another consequence of the unexamined grayscale assumption (§3.1).

**Risk level: low, conditional on §3.1.**

### 3.16 `display()` and Colab-only calls
Cell 18 calls `display(...)`; cells write to `/content/...`. Any port to a script must strip these.

**Risk level: low.**

### 3.17 Evaluation metrics computed on clamped predictions only
Model output is clamped inside `forward`, so evaluation never sees the raw estimate. Since KLA
scores exactly what we save, clamped evaluation is the *correct* thing to report — but we must also
know how often clamping is active, otherwise a systematically over-shooting model looks fine on
PSNR while wasting capacity.

**Risk level: low.** Mitigation: log clamp-activation rate as a diagnostic.

---

## 4. What violates reproducibility requirements

| Violation | Where | Required fix |
|---|---|---|
| Unpinned dependencies | Cell 1 | `requirements.txt` with exact versions, torch index documented |
| Implicit Colab torch version | environment | Pin, and record actual versions in the run snapshot |
| No config snapshot per run | Cell 18 | Write `config.json` next to each checkpoint |
| No experiment CSV | Cell 18 | `results/experiments.csv` with the mandated columns |
| DataLoader worker seeding | Cell 10 | `generator=` + `worker_init_fn=` |
| No `torch.use_deterministic_algorithms` option | Cell 2 | Strict-determinism flag |
| Checkpoint lacks architecture config | Cell 18 | Self-describing checkpoint |
| Absolute Colab paths | Cells 4, 18, 23 | CLI arguments with sane defaults |
| No clean-environment test | — | `scripts/smoke_test.sh` |
| No resume | Cell 18 | `--resume` from `last.pth` |
| Metrics not persisted per image | Cell 18 | Per-image CSV, mean **and** std |

---

## 5. What must be corrected (ordered by severity)

1. **Define and implement the inference output-size contract.** Without GT at test time the pipeline
   must derive the output size from the input. Declared assumption in §7.A1.
2. **Remove the silent grayscale collapse.** Support 1 and 3 channels explicitly; detect from data.
3. **Fix the epoch-invariant sampling** so augmentation is a real distribution
   (`samples_per_image` + epoch-aware seeds), while validation stays frozen.
4. **Make dtype-driven normalization deterministic and logged**, decided before channel math.
5. **Hard-fail (or loudly log) missing official pairs** instead of silently substituting synthetic
   samples into validation.
6. **Pad-to-multiple-of-16 at the model boundary**, crop back to the exact target size; add shape
   tests including odd dimensions.
7. **Move clamping out of training** (flagged), clamp at inference/eval.
8. **Self-describing checkpoints** with embedded config + format version.
9. **LPIPS strictly evaluation-only**; `inference.py` must not import it.
10. **Pin every dependency**; add `scripts/smoke_test.sh` that proves a cold install works.
11. **Persist experiments** to `results/experiments.csv` with seeds and full configuration.
12. **Modern AMP API**, seeded workers, optional strict determinism.
13. **Calibrate degradation ranges from the official paired data** instead of inventing them.

---

## 6. What can remain unchanged

- The residual-on-bicubic formulation (§1.1). Keep.
- Encoder widths 32/64/128/256 with a 512 bottleneck, Conv→GroupNorm→GELU blocks. Keep as the
  frozen baseline architecture; no transformers, no GANs, no diffusion.
- Charbonnier + 0.2·(1 − SSIM) as the *initial* loss configuration, pending the loss ablation.
- AdamW, lr 2e-4, weight decay 1e-5, cosine schedule, grad-clip 1.0, AMP on CUDA.
- The six-permutation randomized degradation order.
- Three mechanisms only: additive Gaussian, multiplicative speckle, downsampling.
- The overfit-two-samples sanity gate.
- PSNR / SSIM / LPIPS / MAE as the metric set.
- Best-validation-PSNR checkpoint selection (with LPIPS and SSIM recorded alongside).

The notebook's *judgement* is good. Its *engineering* is what needs replacing.

---

## 7. Assumptions

Every item below is something the official document does **not** settle. Each is isolated in
configuration, defaulted to the most defensible reading, and stated in the README.

### A. Contract assumptions (highest impact)

**A1. Output size at inference.** The document says outputs must be "at the expected ground-truth
resolution" and that evaluation images are "approximately 256×256 or 512×512", but hidden test data
ships **degraded inputs only**. Assumption: the output size is `input_size × scale_factor`, where
`scale_factor` defaults to a configured integer (initially 2) and is overridable via
`--scale`. Additionally supported and documented: `--target-size H W` for a fixed output size, and
`--size-map FILE` for an explicit per-file mapping. Rationale: an integer upscale is the only rule
derivable from the input alone; the two documented evaluation sizes (256, 512) are consistent with
128→256 and 256→512 at scale 2. **This assumption is the single largest risk in the submission and
is called out in the README and the PPT.** The scale used is also recorded in the checkpoint so the
training-time and inference-time contracts cannot silently diverge.

**A2. Channel count.** Assumption: the dataset is single-channel grayscale (SEM/inspection imagery
normally is), but the code supports 3-channel and the trained checkpoint records which was used.
If official data proves to be RGB, only the config changes.

**A3. Output file format and naming.** Assumption: same stem as the input, PNG, 8-bit, values
scaled from [0,1] by 255 with round-half-to-even, unless the official dataset indicates 16-bit. A
`--out-ext` and `--bit-depth` flag exist. Rationale: PNG is lossless and the notebook's layout
assumed `.png`; lossy formats would corrupt scoring.

**A4. Value range of saved outputs.** Assumption: clip to [0,1] before quantization, because
"KLA will score the images exactly as saved" and GT is defined on [0,1].

### B. Degradation assumptions

**A5. Speckle form.** `x * (1 + N(0, σ_s))`, i.e. multiplicative Gaussian speckle
(`imnoise`-compatible). Configurable.

**A6. Gaussian noise form.** Additive i.i.d. zero-mean `N(0, σ_g)` applied in normalized units.

**A7. Downsampling kernel.** `area` averaging by default (the standard antialiased choice, and what
a real detector binning operation resembles), with `bicubic` and `bilinear` selectable. The official
kernel is not disclosed, so training samples across kernels for robustness.

**A8. Noise-level ranges.** The notebook's `σ_g ∈ [0.005, 0.08]`, `σ_s ∈ [0.01, 0.15]`,
`scale ∈ {2, 4}` are **unsourced guesses**. Assumption: use them as the initial prior, then
recalibrate from the official paired data by estimating residual statistics; the calibration
procedure and its outcome are recorded in `docs/EXPERIMENT_LOG.md`. Ranges live in
`configs/degradation.yaml`.

**A9. Noise applied in normalized units** ([0,1] scale) rather than 8-bit units.

**A10. Independent noise realizations.** Each (source, seed) pair draws fresh noise; no noise array
is reused across samples. Enforced by a per-sample RNG derived from a master seed, source id and
sample index.

### C. Split and evaluation assumptions

**A11. Splits.** 80/10/10 at source-image level with a fixed seed, as in the notebook. The local
test split is only ever used for final reporting, never for selection. The hidden KLA test set is
never trained on.

**A12. Model selection metric.** Validation PSNR primarily, with SSIM and LPIPS recorded every
epoch; if two configurations are within noise on PSNR the tie is broken by LPIPS. Declared because
KLA's exact weighting is undisclosed.

**A13. Determinism scope.** Bit-exact on CPU with a fixed seed; on CUDA, deterministic within a
fixed hardware/driver/library combination when strict mode is enabled, and statistically
reproducible otherwise. Stated honestly rather than overclaimed.

### D. Environment assumptions

**A14. Evaluator hardware.** NVIDIA H100, CUDA available. CPU fallback is implemented and tested,
but throughput numbers are reported per measured device with the device recorded.

**A15. No network at inference.** `inference.py` imports only torch, numpy and Pillow. LPIPS,
scikit-image, matplotlib, pandas and PyYAML are evaluation/training-only.

**A16. Batch processing.** Inputs of identical shape are batched; mixed shapes fall back to
per-image processing. Batch size is a CLI flag defaulting to a value safe on modest GPUs.

---

## 8. Audit conclusion

The notebook is a legitimate starting point and its core design decisions survive the audit:
residual learning on a bicubic base, exactly the three official degradations, randomized ordering,
seeded generation, source-level splitting and the mandated metric set.

Three defects are severe enough to block a submission if carried over unchanged:

1. **No inference-time output-size contract** (§3.12, §2.2) — makes the pipeline unscoreable.
2. **Epoch-invariant sampling** (§3.10) — reduces "augmentation" to a fixed, tiny dataset.
3. **Silent grayscale collapse and heuristic normalization** (§3.1, §3.2) — data corruption risk.

The plan is to preserve the notebook's model and training recipe verbatim as the frozen baseline,
rebuild the surrounding engineering as a tested package, and change one variable at a time with
every experiment recorded in `results/experiments.csv`.
