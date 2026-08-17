#!/usr/bin/env python3
"""Executable KLA restoration acceptance smoke.

It builds a small first-party corpus, trains a deliberately tiny model for one
CPU epoch, runs the public inference CLI, and checks that every input has one
same-name output with the expected x2 dimensions and numeric range.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs" / "smoke_submission")
    args = parser.parse_args(argv)
    work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir
    source_dir = work_dir / "sources"
    corpus_dir = work_dir / "corpus"
    run_root = work_dir / "runs"
    output_dir = work_dir / "restored"
    report_path = work_dir / "smoke_report.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands: list[dict[str, object]] = []
    try:
        commands.append(
            _run([
                python,
                "scripts/generate_clean_sem_sources.py",
                "--out", str(source_dir), "--count", "12", "--size", "128", "--seed", "20260817",
            ])
        )
        commands.append(
            _run([
                python,
                "scripts/materialize_restoration_data.py",
                "--source-dir", str(source_dir), "--out", str(corpus_dir),
                "--seed", "20260817", "--views-per-source", "1", "--crop-size", "128", "--scale", "2",
            ])
        )
        experiment = "smoke_submission"
        commands.append(
            _run([
                python, "train.py", "--config", "configs/baseline.yaml",
                "--gt-dir", str(corpus_dir / "train" / "GT"),
                "--noisy-dir", str(corpus_dir / "train" / "NoisyLR"),
                "--experiment-id", experiment, "--epochs", "1",
                "--set", f"paths.output_dir={run_root}",
                "--set", "model.base_channels=4", "--set", "model.depth=2",
                "--set", "data.patch_size=64", "--set", "data.samples_per_image=1",
                "--set", "data.train_mode=official", "--set", "data.eval_mode=official",
                "--set", "train.batch_size=2", "--set", "train.num_workers=0",
                "--set", "train.amp=false", "--set", "train.eval_lpips=false",
            ])
        )
        checkpoint = run_root / experiment / "best.pth"
        test_input = corpus_dir / "test" / "NoisyLR"
        commands.append(
            _run([
                python, "inference.py", "--input_dir", str(test_input),
                "--output_dir", str(output_dir), "--checkpoint", str(checkpoint),
                "--scale", "2", "--out-ext", ".png", "--report", str(work_dir / "inference_report.json"),
            ])
        )
        inputs = sorted(test_input.glob("*.npy"))
        outputs = sorted(output_dir.glob("*.png"))
        if not inputs or len(inputs) != len(outputs):
            raise AssertionError(f"input/output count mismatch: {len(inputs)} vs {len(outputs)}")
        for source, restored in zip(inputs, outputs):
            if source.stem != restored.stem:
                raise AssertionError(f"stem mismatch: {source.name} != {restored.name}")
            source_array = np.load(source)
            output_array = np.asarray(Image.open(restored), dtype=np.float32) / 255.0
            if output_array.shape[:2] != (source_array.shape[0] * 2, source_array.shape[1] * 2):
                raise AssertionError(f"unexpected restored size: {output_array.shape} from {source_array.shape}")
            if not np.isfinite(output_array).all() or output_array.min() < 0 or output_array.max() > 1:
                raise AssertionError("restored output is not finite and within [0, 1]")
        report = {
            "train_exit_code": 0,
            "inference_exit_code": 0,
            "inputs": len(inputs),
            "outputs": len(outputs),
            "commands": commands,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        failure = {"train_exit_code": None, "inference_exit_code": None, "commands": commands, "error": str(exc)}
        report_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
