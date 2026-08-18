# Curated real SEM structure sources

These are clean, artifact-free SEM structure renders (DRAM 6F2 folded-bitline
cell arrays and FinFET fin/gate/contact arrays) curated from the SEMICON India
2026 "Drift-Sense" dataset drop. They are the ground-truth-quality source images
used, together with a larger first-party synthetic source set, to build the
KLA-faithful GT/NoisyLR restoration corpus.

Curation (see `scripts/curate_sem_sources.py`) keeps only clean structures:
gallery reference crops, full reference patches, un-annotated search
ground-truth renders, and composite layer-stack grayscales. It excludes charts,
QR codes, annotated overlays, pre-degraded artifact demos, montages and tiny
polygon thumbnails.

Downstream:

* `scripts/materialize_restoration_data.py` turns these into paired GT/NoisyLR
  using only the three KLA-disclosed degradations (additive Gaussian noise,
  multiplicative speckle, downsampling), split source-disjointly by SHA-256.
* `configs/submission_robust.yaml` trains on that corpus and additionally enables
  the extended SEM acquisition-artifact augmentation (`src/kla_restore/extended_degradation.py`)
  so the model gains tolerance to beam-spot blur/astigmatism, Poisson shot noise,
  detector readout noise, vignetting, gamma miscalibration, barrel distortion,
  charging streaks and raster drift/jitter.
