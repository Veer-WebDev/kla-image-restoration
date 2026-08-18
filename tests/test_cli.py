from __future__ import annotations

from pathlib import Path

from inference import output_path_for


def test_installed_console_entry_points_expose_help_parsers() -> None:
    from kla_restore import cli

    assert callable(cli.train_main)


def test_inference_preserves_relative_input_name_and_extension_by_default(tmp_path: Path) -> None:
    input_dir = tmp_path / "NoisyLR"
    nested = input_dir / "lot_a"
    nested.mkdir(parents=True)
    source = nested / "wafer_01.tiff"
    source.touch()

    assert output_path_for(source, input_dir, tmp_path / "restored") == (
        tmp_path / "restored" / "lot_a" / "wafer_01.tiff"
    )


def test_inference_output_name_changes_only_when_explicitly_requested(tmp_path: Path) -> None:
    input_dir = tmp_path / "NoisyLR"
    input_dir.mkdir()
    source = input_dir / "wafer_01.png"
    source.touch()

    assert output_path_for(
        source, input_dir, tmp_path / "restored", suffix="_restored", out_ext=".tif"
    ) == (tmp_path / "restored" / "wafer_01_restored.tif")
