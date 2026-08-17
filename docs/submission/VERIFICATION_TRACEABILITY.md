# Drift-Sense Submission Verification Traceability

**Date:** 2026-08-17  
**Scope:** the current committed Drift-Sense localization solution.  
**Evidence boundary:** all measured image data below are synthetic. The official
KLA/Applied Materials evaluation set and an accessible live Colab runtime were
not available in this session. This document does not represent synthetic results
as official acceptance results.

## Requirement-to-check map

| Requirement / changed public output | Concrete check actually run | Observed result | Evidence / status |
|---|---|---|---|
| Official Python inference interface takes Reference + Search and prints `(x,y)` | `python infer.py --reference data/drift_sense_space/output/test/reference/00000.png --search data/drift_sense_space/output/test/search/00000.png` | Printed `115.03,567.38`; paired fixture ground truth is `115.9,567.4`, error **0.870 px** | Direct public-interface result on a held-out external synthetic pair. Representative, not official hidden acceptance. |
| Core matcher and generator behavior | `python -m pytest -q` | **10 passed** | Covers unique localization, subpixel output, ambiguity, official center tie-break, rotation-aware geometric path, generator, invalid architecture, and RGB. |
| Dataset generator creates correctly labelled Reference/Search pairs | `generate_dataset.py` then `evaluate.py` executed in the preceding smoke loop | Generator wrote a manifest and evaluator consumed it, including the new feature-scale controls | Direct integration boundary exercised. The scale-stress smoke also confirmed the result is a difficult geometric scenario, not a fabricated gain. |
| Official tie rule for multiple matches | Matcher default is `center_tiebreak=True`; 200-case evaluator run used that default | `test_big`, n=200: success@10 **59.0%**, median 1.33 px, 209 ms/sample; unique 92 cases: 97.8% @10; ambiguous 108: 25.9% @10 | `results/localize_test_big_current.json`. The external fixture has arbitrary crop labels, which conflicts with the task's nearest-center convention for ties. |
| 1px and 5px rubric reporting | Evaluator's current JSON emitted `positive_pair_confusion` at 1 and 5px in the 200-case run | @1px TP=75, FN=125, 37.5%; @5px TP=118, FN=82, 59.0% | TN/FP are explicitly `null` because ordinary manifests contain only positive localization pairs. No synthetic negatives were invented. |
| Search-noise robustness and PR-vs-noise request | `analysis/noise_sweep.py --levels 0 0.3 0.6 --calibration-n 30 --test-n 30 --seed 777` | Separate calibration/test PR result ledger produced for all levels | `results/noise_sweep_seed777.json` and `.svg`. Heavy noise made score thresholding unreliable, disclosed rather than hidden. |
| Slide-required charging, distortion, rotation, and ±20% polygon/feature scale | Generator exposes `--charging-*`, `--barrel-k`, `--rotation-max-deg`, and `--feature-scale-min 0.8 --feature-scale-max 1.2` | Implemented and smoke-exercised through generator/evaluator. Rotation and radial distortion remain an open robustness gap. | `generate_dataset.py`, `generator.py`, README and architecture PDF. No false robustness claim. |
| RGB optical bonus | RGB generator test within full pytest suite | 3-channel pair generation and luminance localization test passed | Synthetic only. |
| Alternative model exploration | `analysis/feature_baseline.py --manifest .../test/manifest.csv --limit 40` | SIFT + BF matching + RANSAC: **0.0%** @5/@10, median 411.31px, 260ms/sample, median 0 inliers | `results/feature_sift_test40.json`. Rejected from deployment. |
| Architecture PDF | MiKTeX `pdflatex` run twice after edits; `findstr Overfull model_architecture.log` produced no match | 9-page PDF compiled, no overfull-box warning | `docs/submission/model_architecture.pdf`. Opened in Chrome for manual inspection. |
| Colab reproducibility | `json.load(open('notebooks/colab_smoke.ipynb'))`, assert nbformat 4 and 7 cells | `COLAB_NOTEBOOK_VALID cells=7` | Upload-based CPU smoke notebook is committed. Live Colab execution is externally blocked because no runtime was available. |

## Acceptance assessment

The end-user command path, generation-to-evaluation integration, regression
suite, result-ledger pipeline, architecture PDF, and Colab notebook structure
were directly observed to work. The project is therefore **workflow-validated
on representative synthetic data**.

Actual hackathon acceptance is still externally blocked by the unavailable
official test set. The submitted documents distinguish this boundary throughout.

## Commands to repeat

```cmd
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe infer.py --reference REF.png --search SEARCH.png
.venv\Scripts\python.exe evaluate.py --manifest PATH\manifest.csv --cm-thresholds 1 5 --json-out report.json
.venv\Scripts\python.exe analysis\noise_sweep.py --out results\noise_sweep --levels 0 0.3 0.6 --calibration-n 30 --test-n 30 --seed 777
```
