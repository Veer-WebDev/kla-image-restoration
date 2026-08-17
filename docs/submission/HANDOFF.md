# Drift-Sense Localization - Handoff to Codex

Full context for continuing this task in Codex (or any agent). Pairs with
`CODEX_MASTER_PROMPT.md` (the prompt to paste) and `RUN_REPORT.md` (decision
tree + results).

## 1. Task (from official Applied Materials PDF + HF Space slides)
Predict the center (x,y) of a high-res **Reference** (1000x1000 @1nm/px, 1um FOV)
inside a degraded **Search** (1000x1000 @10nm/px, 10um FOV). Reference appears
shrunk 10x somewhere in Search. Metric = Euclidean pixel error. If several tiles
match, report the one **closest to the Search-image center**.

Scoring rubric: 50% inference (coords + compute time, >=30 test cases, CM at
1-5px), 30% augmentation/generator code (literature-cited SEM realism), 10%
failure explainability, bonus for RGB optical-tool images.

Slides also require these augmentations: distortion, charging, rotation 1-3 deg,
polygon scaling -20%..+20%; and a precision-recall vs noise study.

## 2. Repo layout (branch master; all work committed)
```
infer.py               official CLI: --reference --search [--verify] -> prints "x,y"
evaluate.py            manifest -> error stats + success@{2,5,10,20}px + unique/ambiguous split
generate_dataset.py    CLI: --out --n --seed --arch{dram,finfet,mixed} --search-speckle --search-readout --rgb
src/drift_localize/
  __init__.py          exports predict, LocalizeResult, generator
  matcher.py           NCC localizer (see below)
  generator.py         synthetic DRAM/FinFET SEM generator (see below)
tests/test_matcher.py  9 tests, all pass
results/experiments.csv  run ledger (append every run)
docs/EXTERNAL_RESOURCES.md  provenance + citations
docs/submission/model_architecture.{tex,pdf}  8-page explainer
docs/submission/RUN_REPORT.md  decision tree + results
data/                  GITIGNORED. External splits at data/drift_sense_space/output/{train,val,test,test_big}/manifest.csv
```

## 3. matcher.py behavior
- Reads both images grayscale (so RGB input works via luminance).
- Multi-scale templates (DEFAULT_SCALES 9,9.5,10,10.5,11), `cv2.matchTemplate`
  TM_CCOEFF_NORMED, top-K peaks with NMS (`_iter_peaks`), parabolic subpixel
  (`_parabolic_subpixel`).
- `LocalizeResult(x, y, score, scale, n_tied_peaks, ambiguous)`;
  `ambiguous = n_tied>1` within `tie_margin=0.03`.
- `center_tiebreak=True` (default): among tied peaks pick the one nearest the
  Search center (official rule). `False` = raw argmax.
- `verify=False` default: optional fine re-rank, measured to add nothing.
- **`angles=DEFAULT_ANGLES=(0.0,)`: opt-in rotation search. CURRENTLY BROKEN** -
  when angles != (0,), it switches to masked TM_CCORR_NORMED for ALL angles so
  scores are comparable, but CCORR_NORMED is a weaker discriminator than CCOEFF,
  so accuracy drops (measured 20%->5% on 3-deg rotated data). Fix by computing a
  proper masked zero-mean NCC, or use a different rotation strategy. Default
  (0,) path is unchanged and safe.

## 4. generator.py behavior
- `generate_sample(architecture, rng, *, search_speckle_sigma, search_readout_sigma,
  zoned=True, rgb=False, charging_prob, charging_intensity, barrel_k,
  rotation_max_deg)` -> dict(reference_img, search_img, gt_x, gt_y, architecture).
- Builds 10000x10000 fine canvas @1nm/px, random 1000x1000 crop = Reference,
  10x downsample = Search. GT center = crop center in Search px.
- `zoned=True`: array "mats" + irregularly-spaced "strips" + random fiducial
  "constellation" (density ~1 per (400nm)^2) -> locally-unique landmarks. This
  was ESSENTIAL: a purely periodic canvas scored ~0% (correctly - no anchor).
- SEM noise (all literature-tied): `_edge_brighten` (SE edge effect),
  Poisson shot (dose), Gaussian readout (independent per capture), optional
  speckle. New geometric augs: `_charging_streaks`, `_barrel_distortion`,
  `_rotate_small`.
- `_to_optical_rgb`: intensity->color LUT for the RGB bonus (3-channel BGR).
- **TODO from slides:** polygon scaling +/-20% (global CD/linewidth bias). The
  per-line width jitter is +/-10% fixed in `_stripe_mask`; add a controllable
  global scale param.

## 5. Measured results (SYNTHETIC, honest, never claim as official)
External Drift-Sense test_big (200 samples):
| method | success@10px | median | time |
|---|---|---|---|
| NCC default | 75.5% | 1.03px | 156ms |
| NCC +verify | 75.5% | 1.03px | 470ms |
| unique-peak subset | 97.8% | 0.83px | (n=92) |
| ambiguous subset | 56.5% | 1.41px | (n=108) |

Own-generator noise robustness (30 samples each):
| speckle | success@5px | confident subset | ambiguous subset |
|---|---|---|---|
| 0.0 | 86.7% | 22/30 @100% | 8/30 @50% |
| 0.3 | 60.0% | 14/30 @100% | 16/30 @31% |
| 0.6 | 36.7% | 5/30 @100% | 25/30 @24% |

Augmentation impact (20 samples, own generator, no rotation search):
| aug | success@5px | median |
|---|---|---|
| clean | 70% | 0.09px |
| charging | 60% | 0.09px |
| barrel k=0.05 | 20% | 416px |
| rotation +/-3deg | 20% | 16.9px |

**Central finding:** the ~25% (and noise/rotation) failures split into a
resolvable population NCC nails to ~0.05-1px, and an appearance-ambiguous
population (periodic arrays / heavy noise burying fiducials) that a SINGLE
Search image physically cannot disambiguate. The ambiguity flag predicts this
at ~98% precision. Geometric distortion (rotation/barrel) is the one failure
that is NOT a hard ceiling and SHOULD be fixable (objective 1).

## 6. Honesty / reproducibility rules (carried from user)
- Never fabricate; label all metrics synthetic; never claim official KLA/AMAT
  numbers; never tune on hidden test. Fixed seeds, disjoint splits.
- Inference: numpy+opencv only, CPU, no internet, no GPU, no DL dependency.
- Simplicity-first: justify any complexity with real numbers vs baseline.
- License: the external HF Space generator is a measurement fixture only, NOT
  vendored; our generator is a clean reimplementation. No proprietary fab data.
- Commit as you go; run pytest before claiming done.

## 7. Citations to fill into EXTERNAL_RESOURCES.md (from slides)
IRDS 2017 More Moore; IRDS 2024 More Moore; ITRS 2015 More Moore; IBM Research
"Opportunities and Challenges of FinFET ... 14nm"; Semiconductor Engineering
"7nm Fab Challenges"; FreePDK15 predictive PDK (arXiv 2009.04600); arXiv
2007.14448 (NC-FinFET); TI patent EP0780901A2 (arcuate moats/wavy bitlines);
EE Times "Hynix DRAM layout"; US Patent 5,554,874 (6T SRAM cell); US Patent
6,938,226 (7-track standard cell); imec semi-damascene / logic roadmap; IBM
BEOL Cu interconnects. (TSV family had no verified source - flag for own cite.)
Existing named sources already in file: Reimer (SEM), Janesick (Photon
Transfer), Goodman (Speckle).

## 8. Git state at handoff
Latest commits (newest first):
- cd00e7f  slide augmentations + opt-in rotation search (WIP, rotation broken)
- f9aa506  RGB optical bonus
- 35f7013  PDF: generator+robustness+tie-break sections (8pp)
- 0f9175a  docs: generator/robustness/tie-break
- d173fb2  robustness sweep ledger
- 7ee7643  tracked generator + CLI + tests
- fabece3  center tie-break
- c6387f9  run report
Backups: ~/.jcode/openai-auth.json.bak-relogin (pre-relogin).

## 9. Tooling notes (Windows)
- pytest: `.venv\Scripts\python.exe -m pytest -q` (9 tests).
- PDF: `"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" model_architecture.tex` run TWICE.
- cmd has no tail/head; use findstr / more +N. Multiline `python -c` swallows
  stdout - write a temp .py. Long commands (>120s) must run in background.
