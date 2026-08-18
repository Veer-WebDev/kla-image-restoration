from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.materialize_restoration_data import materialize


def test_materialize_source_split_and_three_degradation_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        Image.new("L", (128, 128), color=index * 20).save(source / f"ref_{index}.png")

    summary = materialize(
        source_dir=source,
        out_dir=tmp_path / "out",
        seed=7,
        views_per_source=6,
        crop_size=128,
        scale=2,
        split_ratios=(0.5, 0.25, 0.25),
    )

    assert summary["n_sources"] == 6
    assert summary["source_sets_disjoint"] is True
    assert set(summary["orders"]) == {"GSD", "GDS", "SGD", "SDG", "DGS", "DSG"}
    assert (tmp_path / "out" / "train_manifest.csv").exists()
    assert (tmp_path / "out" / "dataset_card.json").exists()

    split_by_source: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        with (tmp_path / "out" / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            split_by_source.setdefault(row["source_sha256"], set()).add(split)
            assert (tmp_path / "out" / row["gt_path"]).is_file()
            assert (tmp_path / "out" / row["noisylr_path"]).is_file()
            assert row["order"] in {"GSD", "GDS", "SGD", "SDG", "DGS", "DSG"}

    assert all(len(splits) == 1 for splits in split_by_source.values())
