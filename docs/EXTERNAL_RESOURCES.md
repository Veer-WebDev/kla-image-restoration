# External Resources and Data Provenance

The official KLA problem statement permits public external data and pretrained weights only when their licences allow competition use and all sources are disclosed. This is the authoritative disclosure log for this repository.

## Used by the current baseline

| Resource | Role | Licence / source | Status |
|---|---|---|---|
| PyTorch / torchvision | Training and inference framework | BSD-style licence. https://pytorch.org/ | Used, declared in `requirements.txt`. |
| LPIPS (AlexNet metric weights) | Evaluation metric only | LPIPS package, loaded lazily. https://github.com/richzhang/PerceptualSimilarity | Used only in recorded evaluation where weights were available. Never imported by inference. |
| `scripts/make_fixtures.py` output | Local deterministic pipeline fixture and synthetic sanity data | Repository source | Used for tests and the committed synthetic Colab run. It is not KLA data. |

## Supplied Hugging Face link: not approved for training

- **Name:** Drift-Sense Synthetic Dataset Generator
- **URL:** https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data
- **Revision inspected:** `17a728af3ed6a3ccd44f1d3bab95c525efab847a`, inspected 2026-08-16.
- **Purpose described by source:** synthetic Reference/Search pairs for the separate Applied Materials Drift-Sense problem. The README describes 1000×1000 reference/search imagery, not KLA NoisyLR-to-GT restoration pairs.
- **Licence:** no `LICENSE` file or explicit licence declaration was exposed in the Space file manifest or README at the inspected revision.
- **Current decision:** **not downloaded, not used for training, not included in any metric or checkpoint.**

This avoids two unjustified claims: that the resource is legally usable for competition, and that its Reference/Search target is compatible with the KLA restoration target. It may be reconsidered only after the author provides a suitable licence and a controlled experiment shows a benefit without weakening KLA-data validation.

## Admission checklist for Kaggle or other datasets

Before an external source enters training, add a row with:

1. dataset/model name and immutable URL or revision;
2. licence and confirmation it allows hackathon/competition use;
3. dataset/model card and any attribution requirement;
4. image modality and why it is relevant to semiconductor restoration;
5. exact preprocessing and whether it is pretraining, synthetic GT source or validation-only;
6. source-level split policy proving no overlap with KLA validation/test;
7. a registered experiment ID in `results/experiments.csv`.

No Kaggle dataset is currently admitted because none has been supplied with a licence and dataset card. This is intentional, not a missing benchmark result.
