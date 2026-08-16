from __future__ import annotations

import pytest
import torch

from kla_restore.model import ModelConfig, build_model


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
