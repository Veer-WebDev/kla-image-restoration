# External Resources and Data Provenance

This is the authoritative disclosure log for external code, data and weights
used in this repository.

## Runtime dependencies

| Resource | Role | Licence / source | Status |
|---|---|---|---|
| NumPy | Array math | BSD. https://numpy.org/ | Used, declared in `requirements.txt`. |
| OpenCV (opencv-python-headless) | Image IO, `matchTemplate`, resize | Apache-2.0. https://opencv.org/ | Used, declared in `requirements.txt`. |

The solver has no deep-learning dependency and requires no network access or
GPU at inference time.

## Our own synthetic data generator

`src/drift_localize/generator.py` is a **self-contained reimplementation** that
produces DRAM- and FinFET-style Reference/Search pairs with ground-truth centres.
It is built only from publicly known structural characteristics and standard SEM
imaging models. No proprietary fab data and no code from any external Space are
used. Noise/imaging choices and their public sources:

| Effect | Model | Public source |
|---|---|---|
| Edge brightening | Add gradient magnitude at feature sidewalls | Reimer, *Scanning Electron Microscopy* (Springer, 2nd ed.), SE contrast chapter |
| Shot noise (dose) | Poisson counts scaled by dose | Janesick, *Photon Transfer* (SPIE, 2007) |
| Detector/readout noise | Additive Gaussian, independent per capture | Janesick, *Photon Transfer* (SPIE, 2007) |
| Speckle (robustness) | Multiplicative `img*(1+N(0,σ))` | Goodman, *Speckle Phenomena in Optics* (2007) |
| DRAM / FinFET layout | Word/bit lines + contacts; fins + gates | [IRDS 2024 More Moore](https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf); [IBM, *Opportunities and challenges of FinFET...*](https://research.ibm.com/publications/opportunities-and-challenges-of-finfet-as-a-device-structure-candidate-for-14nm-node-cmos-technology); [EE Times, *Hynix DRAM layout*](https://www.eetimes.com/hynix-dram-layout-process-integration-adapt-to-change/) |

These are the citations to expand in the final presentation (the task requires
2–3 credible public sources per augmentation/noise choice).

## Drift-Sense synthetic data generator (external, reference only)

- **Name:** Drift-Sense Synthetic Dataset Generator
- **URL:** https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data
- **Purpose:** generates synthetic Reference/Search image pairs with ground-truth
  centre coordinates for the Applied Materials Drift-Sense localization task,
  which is exactly the task this repository solves.
- **How it is used here:** run **locally** to produce seeded splits used only to
  *measure* the solver (the `test_big` numbers). The generator code is **not
  vendored or redistributed**. Our own submission uses the independent generator
  above; this Space is a measurement fixture only.
- **Licence caution:** at the inspected revision the Space exposed no explicit
  `LICENSE` file. Because our solver and our generator are clean, independent
  implementations containing **no code copied** from the Space, its licensing
  status does not encumber this repository.

## Honesty notes

- All committed metrics are from **synthetic** Drift-Sense data and are labelled
  synthetic. No metric is claimed as an official KLA/AMAT leaderboard result.
- No parameter is tuned against hidden test data. Splits are seeded and disjoint.
