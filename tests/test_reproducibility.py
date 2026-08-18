from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch

from kla_restore.train import split_keys_by_source_manifest
from kla_restore.utils import derive_seed, seed_everything


def test_seed_everything_replays_python_numpy_and_torch_streams() -> None:
    seed_everything(2026)
    first = (random.random(), np.random.rand(3), torch.rand(3))
    seed_everything(2026)
    second = (random.random(), np.random.rand(3), torch.rand(3))

    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_derived_seeds_are_stable_and_distinguish_sample_identity() -> None:
    assert derive_seed("master", "source-1", 0) == derive_seed("master", "source-1", 0)
    assert derive_seed("master", "source-1", 0) != derive_seed("master", "source-1", 1)


def test_source_manifest_keeps_all_views_of_one_source_in_one_split(tmp_path: Path) -> None:
    manifest = tmp_path / "train_manifest.csv"
    keys: list[str] = []
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source_sha256"])
        writer.writeheader()
        for source_index in range(6):
            source = f"source-{source_index}"
            for view_index in range(2):
                sample_id = f"sample_{source_index}_v{view_index}"
                keys.append(sample_id)
                writer.writerow({"sample_id": sample_id, "source_sha256": source})

    sample_splits, source_splits = split_keys_by_source_manifest(
        keys, manifest, ratios=(0.5, 0.25, 0.25), seed=7
    )
    sample_to_split = {
        sample: split for split, samples in sample_splits.items() for sample in samples
    }
    for source_index in range(6):
        assert len(
            {sample_to_split[f"sample_{source_index}_v{view_index}"] for view_index in range(2)}
        ) == 1
    assert sum(len(values) for values in source_splits.values()) == 6
