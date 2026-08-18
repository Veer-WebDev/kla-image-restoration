# Restoration Corpus Data Card

## Scope

This repository currently contains a **first-party synthetic restoration corpus**.
It is designed to exercise the required KLA input-to-output contract, training
pipeline, degradation model and evaluation tooling. It is not official KLA
training or test data. Its measured results must not be represented as KLA
hidden-test performance.

## Clean-source generation

Clean source images are created locally with
`scripts/generate_clean_sem_sources.py`. The generator uses deterministic
line-space gratings, contact/via arrays, periodic pad grids, Manhattan routing
traces and mixed combinations. These are high-level semiconductor-layout
characteristics, not proprietary layouts or device data.

The supplied Drift-Sense Hugging Face Space is **not used**: at inspected
revision `17a728af3ed6a3ccd44f1d3bab95c525efab847a`, no explicit licence file
or licence declaration was available. The KLA brief requires an external
resource licence that allows competition use. See `docs/EXTERNAL_RESOURCES.md`.

## Pair construction

`python scripts/materialize_restoration_data.py` converts clean sources into
paired `GT` and `NoisyLR` images. It applies exactly the three degradation
families in the KLA brief:

1. additive zero-mean Gaussian noise;
2. multiplicative speckle noise, `x * (1 + N(0, sigma))`;
3. integer downsampling.

Each source receives views in all six operation orders: `GSD`, `GDS`, `SGD`,
`SDG`, `DGS` and `DSG`. NoisyLR values are stored in unclipped `float32 .npy`
files, so the generation boundary does not silently discard out-of-range noise
values. GT is clipped to `[0, 1]` before loss and PNG serialization.

## Split policy and reproducibility

- Split unit: SHA-256 hash of a clean source file, before crop/view generation.
- Splits: deterministic train/validation/test partition from one seed.
- Leakage rule: one source hash can belong to exactly one split.
- Audit trail: each split has a CSV manifest with source hash, crop coordinates,
  degradation order, parameters, seed and relative GT/NoisyLR paths.
- Dataset card: materialization writes `dataset_card.json` alongside manifests.

## Known limitations

The corpus is synthetic and structurally simpler than production semiconductor
inspection imagery. It does not establish performance on official, proprietary
or hidden KLA data. It is an honest reproducibility and pipeline-validation
corpus until licensed, task-compatible data becomes available.
