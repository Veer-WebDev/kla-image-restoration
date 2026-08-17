"""Regression tests for YAML config loading.

These lock down two properties that a silent bug had broken:

* a YAML file must actually override the built-in defaults (the merge result
  has to be *assigned*, not discarded), and
* every shipped config must resolve to a model that builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kla_restore.model import ModelConfig, build_model
from kla_restore.train import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_CONFIGS = sorted(p.name for p in CONFIG_DIR.glob("*.yaml") if p.name != "degradation.yaml")


def test_yaml_file_overrides_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        "experiment_id: tiny_probe\n"
        "model:\n"
        "  name: edsr\n"
        "  base_channels: 8\n"
        "  num_blocks: 2\n",
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    # The whole point of the regression: file values must survive the merge.
    assert config["experiment_id"] == "tiny_probe"
    assert config["model"]["name"] == "edsr"
    assert config["model"]["base_channels"] == 8
    # Untouched defaults must still be present after the merge.
    assert config["model"]["upsample_mode"] == "bicubic"
    assert config["train"]["epochs"] == 40


def test_cli_set_override_wins_over_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text("model:\n  name: nafnet\n", encoding="utf-8")

    config = load_config(cfg_path, overrides=["model.base_channels=12", "train.epochs=3"])

    assert config["model"]["name"] == "nafnet"
    assert config["model"]["base_channels"] == 12
    assert config["train"]["epochs"] == 3


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_every_shipped_config_builds_a_model(name: str) -> None:
    config = load_config(CONFIG_DIR / name)
    model = build_model(ModelConfig.from_dict(config["model"]))
    assert sum(p.numel() for p in model.parameters()) > 0
