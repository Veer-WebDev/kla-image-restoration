# Colab notebooks

## `colab_train.ipynb`

Self-contained GPU training notebook for the KLA image-restoration baseline.

**How to run**

1. Push this repository to a GitHub remote (see repo root instructions).
2. Open the notebook in Colab:
   `https://colab.research.google.com/github/<owner>/kla-image-restoration/blob/master/notebooks/colab_train.ipynb`
   (or `File -> Upload notebook`).
3. `Runtime -> Change runtime type -> GPU (T4)`.
4. Set `REPO_URL` in cell 2 to your remote. For a private repo, use a token URL:
   `https://<TOKEN>@github.com/<owner>/kla-image-restoration.git`.
5. Run all cells.

**What it does**

- Clones the repo and installs dependencies on top of Colab's CUDA PyTorch.
- Generates a synthetic, licence-clean dataset with `scripts/make_fixtures.py`
  (wafer motifs degraded by the project's own deterministic engine).
- Trains the residual U-Net for 40 epochs and logs every metric to
  `results/experiments.csv`, reporting the bicubic baseline each epoch.
- Optionally trains the degradation-aware (FiLM) variant for comparison.
- Runs inference and downloads the checkpoint plus the ledger.

**Honesty note**

All numbers produced here are on synthetic data and are labelled synthetic in the
ledger. They do not describe official KLA performance. The official KLA pairs are
not redistributable, so they are not included. See `docs/EXTERNAL_RESOURCES.md`.
