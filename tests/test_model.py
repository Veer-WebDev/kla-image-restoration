from __future__ import annotations

import pytest
import torch

from kla_restore.model import MODEL_REGISTRY, ModelConfig, build_model, model_summary

ARCHITECTURES = sorted(MODEL_REGISTRY)


@pytest.mark.parametrize("height,width", [(128, 128), (256, 256), (512, 512), (127, 253)])
def test_residual_unet_preserves_requested_output_shape(height: int, width: int) -> None:
    # A narrow instance exercises the same padding, skip and resize implementation
    # while keeping the 512-pixel regression check practical on CPU CI.
    model = build_model(ModelConfig(base_channels=4, depth=4)).eval()
    source = torch.rand(1, 1, height, width)

    with torch.inference_mode():
        restored = model(source, target_size=(height, width), clamp=True)

    assert restored.shape == source.shape
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_model_scale_contract_handles_odd_low_resolution_input() -> None:
    model = build_model(ModelConfig(base_channels=4, depth=4)).eval()
    source = torch.rand(1, 1, 37, 53)

    with torch.inference_mode():
        restored = model(source, scale=2)

    assert restored.shape == (1, 1, 74, 106)


@pytest.mark.parametrize("name", ARCHITECTURES)
@pytest.mark.parametrize("height,width", [(128, 128), (256, 256), (127, 253)])
def test_every_architecture_preserves_output_shape(name: str, height: int, width: int) -> None:
    model = build_model(
        ModelConfig(name=name, base_channels=8, depth=3, num_blocks=3)
    ).eval()
    source = torch.rand(1, 1, height, width)

    with torch.inference_mode():
        restored = model(source, target_size=(height, width), clamp=True)

    assert restored.shape == source.shape
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_every_architecture_honours_scale_contract(name: str) -> None:
    model = build_model(ModelConfig(name=name, base_channels=8, depth=3, num_blocks=3)).eval()
    with torch.inference_mode():
        restored = model(torch.rand(1, 1, 37, 53), scale=2)
    assert restored.shape == (1, 1, 74, 106)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_degradation_aware_variants_consume_a_condition_vector(name: str) -> None:
    model = build_model(
        ModelConfig(
            name=name,
            base_channels=8,
            depth=3,
            num_blocks=3,
            degradation_aware=True,
            condition_dim=4,
        )
    ).eval()
    source = torch.rand(1, 1, 64, 64)
    condition = torch.rand(1, 4)

    with torch.inference_mode():
        restored = model(source, target_size=(64, 64), condition=condition, clamp=True)

    assert restored.shape == source.shape
    with pytest.raises(ValueError):
        model(source, target_size=(64, 64))  # missing condition must fail loudly


def test_unknown_architecture_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="does_not_exist"))


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_return_parts_exposes_bicubic_base_and_residual(name: str) -> None:
    model = build_model(ModelConfig(name=name, base_channels=8, depth=3, num_blocks=3)).eval()
    source = torch.rand(1, 1, 48, 48)
    with torch.inference_mode():
        parts = model(source, target_size=(96, 96), return_parts=True)
    assert set(parts) == {"restored", "base", "residual", "unclamped"}
    for value in parts.values():
        assert value.shape == (1, 1, 96, 96)
    summary = model_summary(model)
    assert summary["params_total"] > 0
