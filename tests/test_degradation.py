from __future__ import annotations

import numpy as np

from kla_restore.degradation import ORDER_PERMUTATIONS, DegradationConfig, degrade, order_matrix, sample_seed


def test_same_seed_and_configuration_reproduce_exact_degradation() -> None:
    clean = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64, 1)
    config = DegradationConfig(scales=(2,), kernels=("area",))
    seed = sample_seed(42, "wafer_001", 5, 2)

    first, first_parameters = degrade(clean, config, seed)
    second, second_parameters = degrade(clean, config, seed)

    assert np.array_equal(first, second)
    assert first_parameters == second_parameters
    assert first.shape == (32, 32, 1)


def test_independent_seed_changes_noise_realization() -> None:
    clean = np.full((32, 32, 1), 0.5, dtype=np.float32)
    config = DegradationConfig(scales=(2,), kernels=("area",))

    first, _ = degrade(clean, config, sample_seed(42, "source", 0, 0))
    second, _ = degrade(clean, config, sample_seed(42, "source", 1, 0))

    assert not np.array_equal(first, second)


def test_all_six_orders_are_available_and_preserve_expected_shape() -> None:
    clean = np.random.default_rng(7).random((48, 64, 1), dtype=np.float32)
    config = DegradationConfig(scales=(2,), kernels=("bicubic",))

    generated = order_matrix(clean, config, seed=123)

    assert tuple(generated) == tuple("->".join(order) for order in ORDER_PERMUTATIONS)
    assert all(image.shape == (24, 32, 1) for image in generated.values())


def test_noisy_values_are_not_clipped_by_default() -> None:
    clean = np.ones((32, 32, 1), dtype=np.float32)
    config = DegradationConfig(
        gaussian_sigma=(0.1, 0.1),
        speckle_sigma=(0.0, 0.0),
        scales=(2,),
        kernels=("area",),
        fixed_order=("gaussian", "speckle", "downsample"),
    )

    noisy, _ = degrade(clean, config, seed=3)

    assert float(noisy.max()) > 1.0
