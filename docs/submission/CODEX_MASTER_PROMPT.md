You are continuing an in-progress, submission-grade solution for the Applied Materials "Drift-Sense" localization task (KLA/AMAT SEMICON India Hackathon 2026). Work in C:\Users\Administrator\Projects\kla-image-restoration (GitHub Veer-WebDev/kla-image-restoration, branch master). Read docs/submission/HANDOFF.md and RUN_REPORT.md FIRST for full context; they are authoritative.

TASK: Given a Reference image (1000x1000 @1nm/px, 1um FOV, high-res) find where it sits inside a degraded Search image (1000x1000 @10nm/px, 10um FOV). Output center (x,y) in Search pixels. Metric = Euclidean pixel error. If multiple tiles match, report the one CLOSEST TO SEARCH-IMAGE CENTER (official rule, already implemented as center_tiebreak=True default).

CURRENT STATE (all committed, 9/9 pytest pass):
- src/drift_localize/matcher.py: classical NCC localizer. Multi-scale (9-11x), top-K peaks+NMS, parabolic subpixel, ambiguity flag (n_tied>1), spec center-tiebreak, optional --verify (off, no gain). OPT-IN rotation search via angles= param EXISTS BUT IS BROKEN: masked TM_CCORR_NORMED underperforms plain TM_CCOEFF_NORMED, so rotated-search scores are not comparable and accuracy DROPS (5% vs 20%). This is the #1 bug to fix.
- src/drift_localize/generator.py: self-contained DRAM/FinFET synthetic generator with literature-cited SEM noise (edge-brighten, Poisson shot, readout Gaussian, speckle) PLUS newly-added charging streaks, barrel distortion, small rotation. RGB optical bonus via rgb=True. Zoned canvas (array mats + irregular strips + fiducial constellation) gives locally-unique landmarks.
- infer.py (CLI: --reference --search [--verify] -> prints x,y), evaluate.py (manifest -> mean/median/p90/success@{2,5,10,20}px + unique/ambiguous split), generate_dataset.py (--out --n --seed --arch --search-speckle --rgb).
- results/experiments.csv is the ledger. docs/submission/model_architecture.tex+pdf (8pp, MiKTeX: "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe", run twice).

KEY MEASURED FINDINGS (synthetic, honest):
- On test_big (200, external Drift-Sense generator): NCC success@10px=75.5%, median 1.03px. UNIQUE-peak subset 97.8% @0.83px; AMBIGUOUS subset 56.5%. The ~25% failures are a FUNDAMENTAL single-image information ceiling on periodic arrays, not a model gap. Ambiguity flag is a 98%-precise confidence signal.
- Own generator: success rises with landmarks. Noise robustness sweep (speckle 0/0.3/0.6): overall success@5px 86.7/60.0/36.7%, but CONFIDENT subset stays 100% @~0.05px at every level (noise shrinks the confident set, doesn't corrupt it).
- Augmentation impact: charging barely hurts (70->60%); barrel & 3-deg rotation BREAK plain NCC (70->20%). Rotation robustness is the main open accuracy problem.

YOUR OBJECTIVES (in priority order):
1. FIX rotation robustness properly. Options: (a) score all angles with the SAME metric (compute masked zero-mean normalized cross-correlation yourself so rotated and 0-deg are comparable), or (b) coarse-to-fine angle search, or (c) log-polar / phase-correlation for rotation+scale estimation, or (d) feature-based (ORB/SIFT + RANSAC affine) as an alternative matcher. Measure each vs the augmented generator; keep what actually wins. Default path must NOT regress the clean 75.5%.
2. Add confusion-matrix reporting at 1px and 5px tolerance to evaluate.py (slide requirement: "CM at 1px-5px accuracy").
3. Add a precision-recall vs noise sweep with a saved matplotlib plot (slide requirement); precision = reported centers within tolerance, recall = true patterns found.
4. Add polygon-scaling +/-20% structural augmentation control to the generator (slide requirement) and a global CD/linewidth bias.
5. Flesh out literature citations in docs/EXTERNAL_RESOURCES.md using the exact list in HANDOFF.md (IRDS 2017/2024, ITRS 2015, IBM FinFET, FreePDK15, TI EP0780901A2, US 5,554,874 / 6,938,226, imec, etc.).
6. Keep the architecture PDF in sync; recompile with MiKTeX twice.
7. Explore alternative models that could beat NCC on the resolvable subset (feature-matching, learned descriptors) but PROVE with numbers vs baseline; do not add DL unless it measurably wins. GPU/Colab is optional and pointless for the info ceiling.

HARD CONSTRAINTS (non-negotiable):
- HONEST + REPRODUCIBLE: never fabricate metrics; label all synthetic; never claim official KLA numbers; never tune on hidden test. Fixed seeds, disjoint splits, log every run to results/experiments.csv.
- Inference stays numpy+opencv only, CPU, no internet, no GPU, no DL dependency at inference. (A learned approach, if it wins, must ship a self-contained lightweight artifact and be justified.)
- Commit as you go with clear messages. Run pytest before claiming done. Prefer the simplest solution that wins on real numbers.
- data/ is gitignored (never commit pixels). The external Drift-Sense splits live in data/drift_sense_space/output/{train,val,test,test_big}/manifest.csv (columns: id,architecture,reference_path,search_path,gt_x,gt_y).
- Windows cmd quirks: no tail/head (use findstr, more +N); multiline python -c stdout is swallowed (write a temp .py); long commands exceed 120s foreground (run in background).

DEFINITION OF DONE for this leg: rotation robustness fixed and measured (clean case not regressed), CM + PR-vs-noise reporting added with a saved plot, polygon-scaling augmentation added, citations filled in, PDF + README + experiments.csv updated, pytest green, all committed. Report a concise before/after metrics table and name the one honest failure mode that remains.

Start by reading docs/submission/HANDOFF.md and RUN_REPORT.md, run pytest to confirm the green baseline, then tackle objective 1.