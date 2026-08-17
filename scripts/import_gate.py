#!/usr/bin/env python3
"""Import + compile gate. Internal dev tool, not a deliverable.

Catches API drift between modules (wrong argument order, renamed helpers,
missing exports) that a plain unit test on one module would miss. Every module
and both mandated entry points are compiled and imported, then a handful of
cross-module signatures are called with real arguments.
"""

from __future__ import annotations

import importlib
import inspect
import py_compile
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODULES = [
    "kla_restore",
    "kla_restore.utils",
    "kla_restore.degradation",
    "kla_restore.model",
    "kla_restore.dataset",
    "kla_restore.metrics",
    "kla_restore.checkpoint",
    "kla_restore.train",
]

ENTRY_POINTS = ["train.py", "inference.py"]

failures: list[str] = []


def check(label: str) -> None:
    print(f"  ok   | {label}")


def fail(label: str, exc: BaseException) -> None:
    failures.append(label)
    print(f"  FAIL | {label}: {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=4)


print("=" * 74)
print("compile")
print("=" * 74)
for rel in ENTRY_POINTS + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*.py"))]:
    path = ROOT / rel
    if not path.exists():
        continue
    try:
        py_compile.compile(str(path), doraise=True, quiet=1)
        check(f"compile {rel}")
    except Exception as exc:  # noqa: BLE001
        fail(f"compile {rel}", exc)

print("=" * 74)
print("import")
print("=" * 74)
for name in MODULES:
    try:
        importlib.import_module(name)
        check(f"import {name}")
    except Exception as exc:  # noqa: BLE001
        fail(f"import {name}", exc)

print("=" * 74)
print("cross-module signatures")
print("=" * 74)
try:
    import numpy as np

    from kla_restore import utils

    # setup_logging(path) then setup_logging(level=...) must both work.
    utils.setup_logging(level="WARNING")
    utils.setup_logging(ROOT / "runs" / "_gate.log", level="WARNING")
    check("setup_logging(path) and setup_logging(level=)")

    tmp = ROOT / "runs" / "_gate"
    tmp.mkdir(parents=True, exist_ok=True)
    utils.write_json(tmp / "x.json", {"a": 1})
    assert (tmp / "x.json").exists()
    check("write_json(path, payload)")

    arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8, 1)
    utils.save_image_float(tmp / "x.png", arr, bit_depth=8)
    assert (tmp / "x.png").exists()
    check("save_image_float(path, array)")

    back = utils.load_image_float(tmp / "x.png", clip=True)
    assert back.array.shape == (8, 8, 1), back.array.shape
    check("load_image_float roundtrip shape")

    assert utils.IMAGE_EXTENSIONS == utils.IMAGE_EXTS
    check("IMAGE_EXTENSIONS alias")

    utils.append_csv_row(tmp / "x.csv", {"a": 1, "b": 2}, ["a", "b"])
    check("append_csv_row(path, row, columns)")
except Exception as exc:  # noqa: BLE001
    fail("utils cross-check", exc)

# Every keyword the training loop passes to build_dataloader / datasets must exist.
try:
    from kla_restore.dataset import DatasetConfig, build_dataloader
    from kla_restore.metrics import LossConfig, compute_metrics
    from kla_restore.model import ModelConfig

    for factory, kwargs in [
        (build_dataloader, {"batch_size", "shuffle", "num_workers", "seed", "pin_memory", "drop_last"}),
        (compute_metrics, {"lpips_model", "device"}),
    ]:
        params = set(inspect.signature(factory).parameters)
        missing = kwargs - params
        assert not missing, f"{factory.__name__} missing {sorted(missing)}"
        check(f"{factory.__name__} accepts {sorted(kwargs)}")

    for cfg_cls, keys in [
        (
            DatasetConfig,
            {
                "patch_size",
                "samples_per_image",
                "synthetic_prob",
                "augment",
                "augment_flips",
                "augment_rot90",
                "channels",
                "training",
                "cache_images",
                "max_eval_size",
            },
        ),
        (ModelConfig, {"in_channels", "out_channels", "base_channels", "depth"}),
        (LossConfig, {"kind", "ssim_weight"}),
    ]:
        fields = set(inspect.signature(cfg_cls).parameters)
        missing = keys - fields
        assert not missing, f"{cfg_cls.__name__} missing {sorted(missing)}"
        check(f"{cfg_cls.__name__} accepts {len(keys)} config keys")
except Exception as exc:  # noqa: BLE001
    fail("config/keyword cross-check", exc)

# The training loop and inference both rely on these being importable and callable.
try:
    from kla_restore.train import CSV_COLUMNS, build_optimizer, build_scheduler, load_config

    cfg = load_config(None, ["train.epochs=3", "data.patch_size=64", "train.amp=false"])
    assert cfg["train"]["epochs"] == 3
    assert cfg["data"]["patch_size"] == 64
    assert cfg["train"]["amp"] is False
    check("load_config with --set overrides (int and bool)")

    from kla_restore.model import build_model

    model = build_model(ModelConfig(base_channels=8, depth=2))
    opt = build_optimizer(model, cfg["train"])
    sched = build_scheduler(opt, cfg["train"], steps_per_epoch=4)
    assert sched is not None
    lrs = []
    for _ in range(12):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[3], f"warmup should raise lr: {lrs[:4]}"
    assert lrs[-1] < lrs[3], f"cosine should decay lr: {lrs[3]} -> {lrs[-1]}"
    check(f"lr schedule warms up then decays ({lrs[0]:.2e} -> {max(lrs):.2e} -> {lrs[-1]:.2e})")
    assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS)), "duplicate CSV columns"
    check(f"CSV_COLUMNS unique ({len(CSV_COLUMNS)} columns)")
except Exception as exc:  # noqa: BLE001
    fail("train.py cross-check", exc)

try:
    import inference

    parser = inference.build_parser()
    args = parser.parse_args(["--input_dir", "a", "--output_dir", "b"])
    assert args.input_dir == "a" and args.output_dir == "b"
    check("inference.py exposes --input_dir/--output_dir")

    size, reason = inference.resolve_target_size((10, 20), "img", target_size=None, size_map=None, scale=2)
    assert size == (20, 40), size
    check(f"resolve_target_size scale contract -> {size} via {reason}")

    size, reason = inference.resolve_target_size(
        (10, 20), "img", target_size=None, size_map={"img": (33, 44)}, scale=2
    )
    assert size == (33, 44), size
    check("resolve_target_size honours size map over scale")
except Exception as exc:  # noqa: BLE001
    fail("inference.py cross-check", exc)

print("=" * 74)
if failures:
    print(f"FAILED {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("import gate: all checks passed")
sys.exit(0)
