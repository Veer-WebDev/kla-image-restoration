# Requirement-to-Evidence Traceability

**Verification date:** 2026-08-16

This document distinguishes observed acceptance-path evidence from synthetic fixture evidence. The official KLA paired images and hidden-test targets are unavailable in this workspace, so an official restoration-quality acceptance result is externally blocked. No synthetic number is labelled an official score.

| KLA requirement / changed public output | Check actually executed | Observed result | Evidence class |
|---|---|---|---|
| Repository installs from its declared package metadata | `python -m ensurepip --upgrade`, then `python -m pip install -e . --no-deps` | Editable wheel `kla_restore-1.0.0-0.editable` built and installed. `pip check` reported `No broken requirements found.` | Installed-package integration |
| Declared `kla-train` console command works after installation | `kla-train --help`, then a one-epoch training run through `kla-train` | The command exposed the same training arguments as `train.py`; ran one CPU epoch, wrote a best checkpoint and split. A missing `kla_restore.cli` module was found by a failing test and repaired before this run. | Installed public interface |
| Standalone training creates a source-level split and paired data flow | Installed command with the 12 deterministic fixture GT/NoisyLR pairs | Pairing reported 12 paired, 0 missing, 0 duplicates; source split was train 10 / val 1 / test 1. | Representative synthetic integration |
| Evaluator-facing inference accepts input and output directories without manual edits | `python inference.py --input_dir ... --output_dir ... --checkpoint ... --scale 2` | Restored 12/12 inputs as 8-bit PNGs on CPU. `_assert_sizes.py` confirmed every 128×128 input produced a 256×256 output with valid dynamic range. Mean end-to-end CPU latency was 29.5 ms/image for this tiny smoke model. | Public acceptance path on fixtures |
| Output-resolution contract handles arbitrary dimensions | `inference.py --target-size 257 259` on a 128×128 input | Saved `odd_case.png` as grayscale PNG of size 259×257. The CLI logged the explicit target-size reason. | Public edge-path integration |
| Paired evaluation measures model against bicubic and records artifacts | `python evaluate.py` using the generated checkpoint and frozen validation split | On the one-image synthetic validation smoke, model PSNR was 17.5611 vs bicubic 17.5565, delta +0.0046 dB; SSIM 0.88880 vs 0.88876. It wrote summary, maps and cost figures. This small result only verifies the evaluation path, not a meaningful quality gain. | Representative synthetic integration |
| Baseline improvement has concrete measured evidence | Existing committed `colab_t4_40ep` result on 80 generated synthetic wafer pairs | Held-out 8-image synthetic test: model PSNR 20.273 vs bicubic 19.978 (+0.296 dB mean), SSIM 0.793 vs 0.688, LPIPS 0.226 vs 0.402, MAE 0.0845 vs 0.0897. PSNR win rate was 37.5%, so the result is not presented as uniformly better. | Measured synthetic benchmark |
| Deterministic degradation, all orders and unbounded NoisyLR behavior | `pytest -q`, `scripts/dev_check.py` | 18 pytest tests passed. Developer checks verified six orders, exact replay for the same seed, different realizations for different seeds, unbounded noisy output, odd shapes, finite loss, gradient flow and exact checkpoint reload. | Unit and component verification |
| Inference does not require LPIPS | CLI imports and executed `inference.py`; optional metric disabled in fixture evaluation | The inference command completed with only torch, NumPy and Pillow imports. LPIPS was explicitly disabled for the fixture evaluator; committed Colab metrics demonstrate LPIPS was available in an evaluation-only environment. | Public interface and dependency boundary |
| Clean-environment reproducibility route | `scripts/smoke_test.sh` added and manually mirrored with the current environment | The actual train → checkpoint → inference sequence passed twice. The script's **fresh virtual-environment dependency installation itself was not executed in this session**, so a fully clean-networked installation remains unverified. | Partially verified |

## Acceptance constraints that remain external

1. The official KLA paired training images are absent. The model cannot be evaluated, calibrated or selected on official validation data.
2. The hidden test has no public GT by design. It must never be used for model selection.
3. The supplied Hugging Face Drift-Sense Space is a different Reference/Search task and has no explicit licence in the inspected source manifest. It remains excluded from training.
4. The recorded 40-epoch Colab evidence used a Tesla T4. KLA will benchmark an H100, so H100 latency remains unmeasured.
5. The default x2 output scaling is an isolated assumption until KLA publishes the exact hidden-test size contract or a size map.

## Verification conclusion

The repository's installed training command, standalone inference command and evaluation command have been exercised across normal and odd-dimension fixture paths. The residual approach has measured synthetic improvements over bicubic on average but also documented PSNR regressions on hard structures. Whether it satisfies KLA's quality requirement cannot be observed without the official paired validation set, and this repository makes that boundary explicit rather than fabricating an acceptance claim.
