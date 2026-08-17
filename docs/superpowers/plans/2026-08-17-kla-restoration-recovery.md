# KLA Restoration Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the prior residual-U-Net restoration package, create a disclosed source-disjoint GT/NoisyLR corpus from the available Drift-Sense Reference imagery using only three permitted degradations, train one reproducible final checkpoint, and assemble a KLA-restoration submission package.

**Architecture:** Keep the current localization implementation only in a tagged archive. On the restoration branch, restore the pre-localization `kla_restore` package from commit `c668591`. Add one materialization script that treats each Drift-Sense Reference PNG as a clean source image, source-splits it before generating views, and uses the existing `DegradationConfig`/`degrade` functions to create paired GT and NoisyLR images. The evaluator-facing entry point is the recovered directory-to-directory `inference.py`; training, evaluation and reporting consume manifests created by the materializer.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, NumPy, Pillow, PyYAML, scikit-image, LPIPS, pandas, matplotlib, pytest, Google Colab GPU, python-pptx.

**Spec:** `docs/superpowers/specs/2026-08-17-kla-restoration-recovery-design.md`

## Global Constraints

- Preserve the current Drift-Sense localization state with an annotated tag before changing the active submission surface.
- Treat `https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data` only as a disclosed source-image collection, never as official KLA restoration data.
- Use only additive Gaussian noise, multiplicative speckle noise and downsampling in the restoration benchmark corpus. Sample all six orders.
- Keep GT normalized to `[0,1]`; do not clip generated NoisyLR before model input.
- Split by source image before crops or degraded views. Use fixed seeds and write manifests.
- Never claim official KLA results, hidden-test performance, or a metric that was not run.
- The final evaluator contract is `python inference.py --input_dir INPUT --output_dir OUTPUT --checkpoint weights/final_model.pth`.
- Include a final checkpoint, resolved config, SHA-256, source/provenance manifest, requirements, measured metrics/runtime and a PPTX.
- Do not overwrite the archive branch/tag. Do not depend on internet during final inference.

---

### Task 1: Preserve localization and restore the historical restoration baseline

**Files:**
- Create: Git tag `archive/drift-sense-localization-20260817`
- Restore: `README.md`, `requirements.txt`, `pyproject.toml`, `train.py`, `inference.py`, `evaluate.py`, `configs/*.yaml`, `src/kla_restore/*.py`, `tests/test_*.py`, `scripts/*.py`, `notebooks/colab_train.ipynb`, `results/colab_t4_40ep/*`, and selected restoration documents from `c668591`
- Preserve: `docs/submission/`, `docs/superpowers/`, `docs/submission/KLA_RESTORATION_COMPLIANCE_AUDIT.md` and current audit/recovery plan history
- Create: `docs/submission/ARCHIVE_NOTICE.md`
- Remove from restoration branch: `infer.py`, `generate_dataset.py`, `src/drift_localize/`, `tests/test_matcher.py`, localization-only result reports and localization architecture documents

**Interfaces:**
- Consumes: clean `master` at `789c127`; Git commit `c668591`.
- Produces: restoration branch with `train.py`, `inference.py`, `evaluate.py`, `kla_restore` package and original regression suite.

- [ ] **Step 1: Create an immutable archive tag**

Run:
```cmd
git status --short
git tag -a archive/drift-sense-localization-20260817 -m "Archive Drift-Sense localization before KLA restoration recovery"
git show --no-patch --decorate archive/drift-sense-localization-20260817
```

Expected: clean worktree before tagging and tag points at the final localization/audit/design history.

- [ ] **Step 2: Create the restoration branch**

Run:
```cmd
git switch -c kla-restoration-submission
```

Expected: active branch is `kla-restoration-submission`; archive tag remains unchanged.

- [ ] **Step 3: Restore historical files without touching Git history**

Run:
```cmd
git restore --source c668591 -- README.md requirements.txt pyproject.toml train.py inference.py evaluate.py configs src/kla_restore tests scripts notebooks/colab_train.ipynb results/colab_t4_40ep docs/AUGMENTATION_JUSTIFICATION.md docs/EXPERIMENT_LOG.md docs/FAILURE_ANALYSIS.md docs/IMPLEMENTATION_AUDIT.md docs/VERIFICATION_TRACEABILITY.md
git rm -r infer.py generate_dataset.py src/drift_localize tests/test_matcher.py
```

Expected: restoration entry points and package are present; the evaluator-facing root has no coordinate-localization script.

- [ ] **Step 4: Add archive notice**

Create `docs/submission/ARCHIVE_NOTICE.md`:
```markdown
# Archived Drift-Sense Localization Work

The tag `archive/drift-sense-localization-20260817` preserves the previous
Reference/Search coordinate-localization prototype. It is not part of the KLA
image-restoration submission branch.
```

- [ ] **Step 5: Run recovered regression suite**

Run:
```cmd
python -m pytest -q
```

Expected: all recovered restoration tests pass. If a test fails due to a missing historic artifact, restore the artifact from `c668591` or change only the test fixture path, never loosen its assertion.

- [ ] **Step 6: Commit baseline recovery**

Run:
```cmd
git add -A
git commit -m "Restore KLA image restoration baseline"
```

### Task 2: Materialize a disclosed source-disjoint restoration corpus

**Files:**
- Create: `scripts/materialize_restoration_data.py`
- Create: `tests/test_materialize_restoration_data.py`
- Modify: `docs/EXTERNAL_RESOURCES.md`
- Modify: `README.md`
- Create: `docs/DATA_CARD.md`

**Interfaces:**
- Consumes: `src/kla_restore.degradation.DegradationConfig`, `sample_seed`, `degrade`; a directory of public-source Reference PNG files.
- Produces: `OUT/{train,val,test}/{GT,NoisyLR}`, `OUT/{train,val,test}_manifest.csv`, `OUT/dataset_card.json`.
- CLI: `python scripts/materialize_restoration_data.py --source-dir DIR --out DIR --seed 20260817 --views-per-source 8 --crop-size 512 --scale 2`.

- [ ] **Step 1: Write failing manifest/source-isolation test**

Create `tests/test_materialize_restoration_data.py`:
```python
from pathlib import Path

from PIL import Image

from scripts.materialize_restoration_data import materialize


def test_materialize_source_split_and_three_degradation_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        Image.new("L", (128, 128), color=index * 20).save(source / f"ref_{index}.png")

    summary = materialize(
        source_dir=source,
        out_dir=tmp_path / "out",
        seed=7,
        views_per_source=1,
        crop_size=128,
        scale=2,
        split_ratios=(0.5, 0.25, 0.25),
    )

    assert summary["n_sources"] == 6
    assert summary["source_sets_disjoint"] is True
    assert set(summary["orders"]) == {"GSD", "GDS", "SGD", "SDG", "DGS", "DSG"}
    assert (tmp_path / "out" / "train_manifest.csv").exists()
```

- [ ] **Step 2: Run the test to verify failure**

Run:
```cmd
python -m pytest tests/test_materialize_restoration_data.py -q
```

Expected: import failure because `materialize_restoration_data` does not yet exist.

- [ ] **Step 3: Implement materialization API and CLI**

Implement:
```python
def materialize(
    *,
    source_dir: Path,
    out_dir: Path,
    seed: int,
    views_per_source: int,
    crop_size: int,
    scale: int,
    split_ratios: tuple[float, float, float],
) -> dict[str, object]:
    """Create source-disjoint GT/NoisyLR directories and auditable manifests."""
```

For each source:

```python
source_id = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
split = source_to_split[source_id]
view_seed = sample_seed(seed, source_id, view_index)
params = sample_params(config, view_seed)
gt = crop_or_resize(load_image_float(source_path, clip=True).array, crop_size, view_seed)
noisylr = degrade(gt, params)
```

Write each manifest row with exactly these columns:

```text
sample_id,source_file,source_sha256,split,view_index,seed,crop_top,crop_left,
gaussian_sigma,speckle_sigma,scale,kernel,order,gt_path,noisylr_path
```

Force all six permutations by assigning `ORDER_PERMUTATIONS[view_index % 6]`
to each view's `fixed_order`. Save GT after clipping to `[0,1]`; save NoisyLR
with a float-preserving `.npy` sidecar and the image representation used by the
recovered loader. The manifest must state how the image file was encoded.

- [ ] **Step 4: Run focused tests**

Run:
```cmd
python -m pytest tests/test_materialize_restoration_data.py tests/test_degradation.py tests/test_dataset.py -q
```

Expected: pass. Confirm one source hash appears in one split only and all six order strings are represented when `views_per_source >= 6`.

- [ ] **Step 5: Add disclosure/data card**

`docs/DATA_CARD.md` must state:

```markdown
- Source imagery: Drift-Sense Synthetic Data Space URL and retrieval date.
- Role: source imagery only; not official KLA restoration data.
- Pair construction: GT crop plus controlled Gaussian/speckle/downsample degradation.
- Split unit: SHA-256 source image hash before crop/view generation.
- Limits: source imagery is synthetic and visually related to the prior localization task; results may not predict KLA hidden-test performance.
```

Add the URL, observed licence information or explicit "no licence file observed" status to `docs/EXTERNAL_RESOURCES.md`.

- [ ] **Step 6: Commit data materialization**

Run:
```cmd
git add scripts/materialize_restoration_data.py tests/test_materialize_restoration_data.py docs/DATA_CARD.md docs/EXTERNAL_RESOURCES.md README.md
git commit -m "Add disclosed restoration data materializer"
```

### Task 3: Validate the train/inference contract before full training

**Files:**
- Modify: `scripts/e2e_smoke.sh` or create `scripts/e2e_smoke.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: materialized `GT` and `NoisyLR`, recovered `train.py`, recovered `inference.py`.
- Produces: a tiny checkpoint and output directory with one restored file per NoisyLR input.

- [ ] **Step 1: Write failing output-directory contract test**

Add to `tests/test_cli.py`:
```python
def test_inference_writes_preserved_stems_for_directory_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "NoisyLR"
    output_dir = tmp_path / "restored"
    input_dir.mkdir()
    make_tiny_image(input_dir / "wafer_01.png")
    checkpoint = make_tiny_checkpoint(tmp_path / "model.pth")

    result = run_inference(input_dir, output_dir, checkpoint)

    assert result.returncode == 0, result.stderr
    assert (output_dir / "wafer_01.png").is_file()
```

- [ ] **Step 2: Run focused test to establish baseline**

Run:
```cmd
python -m pytest tests/test_cli.py -q
```

Expected: pass after restoration; otherwise correct only CLI path/extension compatibility.

- [ ] **Step 3: Implement one clean smoke command**

Create `scripts/e2e_smoke.py` that:

```python
# 1. calls materialize() on six tiny source images
# 2. invokes train.py for one epoch with CPU-compatible settings
# 3. invokes inference.py with --input_dir, --output_dir and --checkpoint
# 4. asserts matching input/output stems, valid dimensions and [0, 1] output range
# 5. writes smoke_report.json with exact commands and elapsed time
```

It must use `subprocess.run(..., check=True, text=True)` and fail on a missing output or mismatched stem.

- [ ] **Step 4: Run complete recovery regression suite**

Run:
```cmd
python -m pytest -q
python scripts/e2e_smoke.py --work-dir runs/smoke_submission
```

Expected: all tests pass and smoke report contains `train_exit_code: 0` and `inference_exit_code: 0`.

- [ ] **Step 5: Commit the executable acceptance path**

Run:
```cmd
git add tests/test_cli.py scripts/e2e_smoke.py README.md
git commit -m "Verify restoration training and inference contract"
```

### Task 4: Produce the fixed training configuration and train one final baseline

**Files:**
- Create: `configs/submission_final.yaml`
- Modify: `train.py` only if recovered CLI cannot accept all configuration values
- Create: `weights/README.md`
- Create after measured run: `weights/final_model.pth`, `weights/final_model.sha256`, `weights/final_model.config.yaml`
- Create after measured run: `results/submission_final/{train_summary.json,history_val.csv,eval_val_summary.json,eval_test_summary.json,environment.txt}`

**Interfaces:**
- Consumes: materialized source-disjoint train/val/test corpus and recovered `kla_restore.train`.
- Produces: one selected `weights/final_model.pth` and non-overwritten run evidence.

- [ ] **Step 1: Define fixed final configuration**

Create `configs/submission_final.yaml` with concrete values:

```yaml
experiment_id: kla_restoration_submission_seed20260817
seed: 20260817
model:
  architecture: residual_unet
  in_channels: 1
  out_channels: 1
  base_channels: 32
  depth: 4
  inference_scale: 2
training:
  epochs: 40
  batch_size: 16
  learning_rate: 0.0002
  num_workers: 2
  device: auto
  loss:
    charbonnier_weight: 1.0
    ssim_weight: 0.1
validation:
  selection_metric: psnr
  split_seed: 20260817
degradation:
  gaussian_sigma: [0.005, 0.08]
  speckle_sigma: [0.01, 0.15]
  scales: [2]
  orders: [GSD, GDS, SGD, SDG, DGS, DSG]
```

Use the exact schema supported by recovered config loading. Do not add unknown YAML keys.

- [ ] **Step 2: Verify the configuration is accepted**

Run:
```cmd
python train.py --config configs/submission_final.yaml --gt-dir DATA\train\GT --noisy-dir DATA\train\NoisyLR --epochs 1 --set training.batch_size=1
```

Expected: writes a checkpoint and resolved config under `runs/kla_restoration_submission_seed20260817/`.

- [ ] **Step 3: Run training on Colab GPU**

Use the Colab notebook task in Task 6. Record device name, CUDA version, PyTorch version, dataset manifest SHA-256, command, start/end UTC, epoch count, selected epoch and all validation rows. Do not choose a checkpoint from the test split.

- [ ] **Step 4: Select and freeze the validation-best checkpoint**

Run:
```cmd
copy runs\kla_restoration_submission_seed20260817\best.pth weights\final_model.pth
copy runs\kla_restoration_submission_seed20260817\resolved_config.yaml weights\final_model.config.yaml
certutil -hashfile weights\final_model.pth SHA256 > weights\final_model.sha256
```

`weights/README.md` must name the exact training data manifest, seed, selected validation epoch, source command and SHA-256.

- [ ] **Step 5: Commit reproducible configuration, not large weights unless portal/repository policy permits**

Run:
```cmd
git add configs/submission_final.yaml weights/README.md
git commit -m "Add fixed restoration submission configuration"
```

Commit the final checkpoint only if repository size policy permits it. Otherwise include exact portal/download instructions plus checksum in `weights/README.md`.

### Task 5: Evaluate the baseline and selected model honestly

**Files:**
- Modify: `evaluate.py` only if the recovered evaluator cannot export all required metrics and runtime fields
- Create: `results/submission_final/METRICS.md`
- Create: `results/submission_final/per_image_metrics.csv`
- Create: `results/submission_final/examples/`
- Create: `results/submission_final/failure_analysis.md`

**Interfaces:**
- Consumes: held-out source-disjoint test GT/NoisyLR, final checkpoint and bicubic baseline.
- Produces: PSNR, SSIM, LPIPS if executable, MAE, per-image results, runtime record and full-resolution visual examples.

- [ ] **Step 1: Write failing result-schema test**

Add a test asserting evaluator JSON includes:

```python
required = {
    "n_images", "model", "baseline", "psnr", "ssim", "mae",
    "ms_per_image_end_to_end", "batch_size", "device", "timing_method"
}
assert required <= report.keys()
```

When LPIPS can execute, assert `"lpips" in report["model"]`; otherwise the report must include `"lpips_status": "unavailable: <actual error>"`.

- [ ] **Step 2: Run test first**

Run:
```cmd
python -m pytest tests/test_metrics.py -q
```

Expected: failure if required report fields are absent.

- [ ] **Step 3: Implement report fields and timing definition**

Measure timing around image discovery, load, preprocessing, tensor transfer, model, synchronization, CPU conversion and save. For CUDA:

```python
if device.type == "cuda":
    torch.cuda.synchronize()
started = time.perf_counter()
# complete end-to-end per-image or per-batch pipeline
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - started) * 1000
```

Store device string, batch size, torch/CUDA versions and command line in output JSON.

- [ ] **Step 4: Evaluate once on held-out test data**

Run:
```cmd
python evaluate.py --gt-dir DATA\test\GT --noisy-dir DATA\test\NoisyLR --checkpoint weights\final_model.pth --output-dir results\submission_final\examples --json-out results\submission_final\eval_test_summary.json
```

Expected: writes baseline and final metrics, per-image rows, restored examples and elapsed end-to-end timing. Do not rerun with changed hyperparameters after inspecting the test result.

- [ ] **Step 5: Write metric/failure documents from the actual JSON**

`METRICS.md` must have a two-row Bicubic vs Residual U-Net table using only values parsed from `eval_test_summary.json`. `failure_analysis.md` must show at least one actual input/bicubic/prediction/GT/error panel and explain only measured failure behavior.

- [ ] **Step 6: Commit evaluation artifacts and tests**

Run:
```cmd
git add evaluate.py tests/test_metrics.py results/submission_final
 git commit -m "Record restoration baseline and final evaluation"
```

### Task 6: Create and execute the Colab reproducibility notebook

**Files:**
- Replace: `notebooks/colab_train.ipynb`
- Create: `notebooks/README.md`
- Create after execution: `results/submission_final/colab_execution.json`

**Interfaces:**
- Consumes: a `git archive` ZIP of `kla-restoration-submission`, the source-image directory or uploaded archive, and the fixed config.
- Produces: trained final checkpoint, evaluation JSON, environment report and an artifact ZIP downloadable from Colab.

- [ ] **Step 1: Add explicit Colab cells**

The notebook must include cells that:

```python
# Cell 1: check GPU and versions
import platform, torch
print({"python": platform.python_version(), "torch": torch.__version__,
       "cuda_available": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "cuda": torch.version.cuda})
```

```python
# Cell 2: upload git archive and optional source-image archive
from google.colab import files
uploaded = files.upload()
```

```python
# Cell 3: install exact dependencies, then materialize/train/evaluate
!python -m pip install -r requirements.txt
!python scripts/materialize_restoration_data.py --source-dir data/source_references --out data/kla_restoration --seed 20260817 --views-per-source 8 --crop-size 512 --scale 2
!python train.py --config configs/submission_final.yaml --gt-dir data/kla_restoration/train/GT --noisy-dir data/kla_restoration/train/NoisyLR
!python evaluate.py --gt-dir data/kla_restoration/test/GT --noisy-dir data/kla_restoration/test/NoisyLR --checkpoint runs/kla_restoration_submission_seed20260817/best.pth --json-out results/submission_final/eval_test_summary.json
```

```python
# Cell 4: archive outputs
!zip -r kla_restoration_artifacts.zip weights results/submission_final runs/kla_restoration_submission_seed20260817
files.download("kla_restoration_artifacts.zip")
```

- [ ] **Step 2: Validate notebook schema locally**

Run:
```cmd
python -c "import json; n=json.load(open('notebooks/colab_train.ipynb', encoding='utf-8')); assert n['nbformat']==4; assert len(n['cells'])>=4; print('COLAB_NOTEBOOK_VALID')"
```

Expected: `COLAB_NOTEBOOK_VALID`.

- [ ] **Step 3: Execute in the user-authorized Colab runtime**

Open the available Colab tab, upload the archive and source-image archive, enable GPU if available, and run cells sequentially. Do not report execution unless actual outputs download or appear in the notebook.

- [ ] **Step 4: Capture run evidence**

Create `results/submission_final/colab_execution.json` from actual notebook outputs:

```json
{
  "executed": true,
  "utc_started": "<observed ISO timestamp>",
  "gpu": "<observed GPU string>",
  "torch": "<observed version>",
  "cuda": "<observed version>",
  "artifact_sha256": "<observed SHA-256>",
  "notes": "Synthetic disclosed source-image corpus; not official KLA data."
}
```

If the user has not granted a working Colab runtime, set `executed` to `false` and record the exact external block. Never create a successful execution record without the actual run.

- [ ] **Step 5: Commit notebook and actual execution evidence**

Run:
```cmd
git add notebooks/colab_train.ipynb notebooks/README.md results/submission_final/colab_execution.json
git commit -m "Add KLA restoration Colab reproduction workflow"
```

### Task 7: Build the required final documentation and solution PPTX

**Files:**
- Modify: `README.md`
- Modify: `docs/EXTERNAL_RESOURCES.md`
- Create: `scripts/verify_submission.py`
- Create: `tests/test_submission_manifest.py`
- Create: `docs/submission/FINAL_SUBMISSION_CHECKLIST.md`
- Create: `docs/submission/SUBMISSION_MANIFEST.json`
- Create: `solution_presentation.pptx`
- Create: `docs/submission/solution_presentation.pdf`

**Interfaces:**
- Consumes: final config, checkpoint hash, manifests, evaluation JSON, examples, failure analysis and runtime report.
- Produces: a 12-slide presentation and a machine-readable submission manifest with only observed values.

- [ ] **Step 1: Write final checklist test**

Create `tests/test_submission_manifest.py` and `scripts/verify_submission.py`. The script accepts `--manifest PATH`, loads JSON, resolves every relative path against repository root, hashes each checksum-declared file with SHA-256, and returns nonzero on a missing file or mismatch. The test must construct a temporary manifest referencing `README.md`, assert exit code zero, then assert exit code nonzero after replacing the declared SHA-256 with `"0" * 64`.

- [ ] **Step 2: Create the PPTX with the supplied 12-slide structure**

Use the `pptx` skill. Populate slides only from real artifacts:

1. title/team/one-line solution;
2. restoration problem and exact input-output contract;
3. source imagery and controlled degradation observations;
4. end-to-end pipeline;
5. preprocessing and three-mechanism data generation;
6. residual U-Net architecture;
7. loss/training configuration;
8. experiment ledger and bicubic baseline comparison;
9. PSNR/SSIM/LPIPS/MAE metrics;
10. runtime/device/batch/timing method;
11. full-resolution examples, actual failure and limitations;
12. reproducibility, external-source disclosure, repository and next steps.

If LPIPS was unavailable, state exactly that and display no LPIPS numeric value.

- [ ] **Step 3: Write final README contract**

README must include executable commands:

```cmd
python -m pip install -r requirements.txt
python scripts/materialize_restoration_data.py --source-dir data/source_references --out data/kla_restoration --seed 20260817 --views-per-source 8 --crop-size 512 --scale 2
python train.py --config configs/submission_final.yaml --gt-dir data/kla_restoration/train/GT --noisy-dir data/kla_restoration/train/NoisyLR
python inference.py --input_dir path\to\NoisyLR --output_dir output --checkpoint weights\final_model.pth
python evaluate.py --gt-dir data/kla_restoration/test/GT --noisy-dir data/kla_restoration/test/NoisyLR --checkpoint weights\final_model.pth --json-out results\submission_final\eval_test_summary.json
```

It must state output stem/format/range, target-size assumption, no-network inference guarantee and synthetic-data evidence boundary.

- [ ] **Step 4: Generate manifest/checklist from artifacts**

`SUBMISSION_MANIFEST.json` includes Git commit, file paths, SHA-256 values, device/software versions, exact result JSON paths and a `data_scope` field:

```json
{
  "data_scope": "Synthetic restoration pairs materialized from disclosed public Drift-Sense Reference imagery; not official KLA training data."
}
```

- [ ] **Step 5: Final clean-environment acceptance run**

Run in a new virtual environment or Colab runtime:

```cmd
python -m pip install -r requirements.txt
python inference.py --input_dir data\klaus_restoration\test\NoisyLR --output_dir submission_smoke --checkpoint weights\final_model.pth --report submission_smoke\report.json
python scripts/verify_submission.py --manifest docs\submission\SUBMISSION_MANIFEST.json
```

Expected: all input stems are written once, no source edits are needed, manifest verification passes and a final report records the actual command/device/runtime.

- [ ] **Step 6: Commit final package**

Run:
```cmd
git add README.md docs solution_presentation.pptx docs/submission/SUBMISSION_MANIFEST.json docs/submission/FINAL_SUBMISSION_CHECKLIST.md weights results
 git commit -m "Package KLA restoration submission artifacts"
```

## Plan self-review

- **Spec coverage:** Task 1 preserves and restores the historical U-Net code. Task 2 creates the disclosed, source-disjoint three-degradation corpus. Tasks 3-5 cover training/inference acceptance and metrics/runtime. Task 6 covers actual Colab execution. Task 7 covers README, weights, presentation and packaging.
- **Placeholder scan:** No implementation step relies on an unspecified dataset role, metric result, file interface or degradation. Values derived from a real run are deliberately written only after that run.
- **Type consistency:** `materialize()` output directories and manifest names feed Task 3, fixed configuration and Task 4. `weights/final_model.pth` is the single checkpoint interface for Tasks 5-7. `inference.py` remains the required evaluator-facing command.
