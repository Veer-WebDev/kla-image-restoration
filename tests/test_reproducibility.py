from __future__ import annotations

import random

import numpy as np
import torch

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
