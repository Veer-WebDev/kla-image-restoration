#!/usr/bin/env bash
# Clean-environment acceptance check: fixture generation -> training -> checkpoint -> inference.
# It deliberately does not need LPIPS or network access once PyTorch and core requirements are installed.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.smoke-venv}"
WORK_DIR="${WORK_DIR:-$ROOT/.smoke-work}"

rm -rf "$VENV_DIR" "$WORK_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip
# Core packages only: inference must not depend on LPIPS/scikit-image.
"$PY" -m pip install 'torch>=2.1,<2.7' 'torchvision>=0.16,<0.22' 'numpy>=1.24,<2' 'Pillow>=10,<12' 'PyYAML>=6,<7' 'tqdm>=4.65,<5'

cd "$ROOT"
"$PY" scripts/make_fixtures.py --out "$WORK_DIR/data" --count 12 --seed 2026
"$PY" train.py \
  --gt-dir "$WORK_DIR/data/GT" \
  --noisy-dir "$WORK_DIR/data/NoisyLR" \
  --config configs/baseline.yaml \
  --experiment-id smoke_clean \
  --epochs 1 \
  --set paths.output_dir="$WORK_DIR/runs" \
  --set model.base_channels=4 \
  --set train.batch_size=2 \
  --set train.num_workers=0 \
  --set train.amp=false \
  --set data.patch_size=64 \
  --set data.samples_per_image=1 \
  --set train.eval_lpips=false

"$PY" inference.py \
  --input_dir "$WORK_DIR/data/NoisyLR" \
  --output_dir "$WORK_DIR/restored" \
  --checkpoint "$WORK_DIR/runs/smoke_clean/best.pth" \
  --scale 2 \
  --report "$WORK_DIR/inference_report.json"

"$PY" - "$WORK_DIR/data/NoisyLR" "$WORK_DIR/restored" <<'PY'
from pathlib import Path
import sys
import numpy as np
from PIL import Image

inputs = sorted(Path(sys.argv[1]).glob("*.png"))
outputs = sorted(Path(sys.argv[2]).glob("*.png"))
assert inputs, "fixture generation produced no inputs"
assert len(inputs) == len(outputs), (len(inputs), len(outputs))
for source, restored in zip(inputs, outputs):
    assert source.stem == restored.stem, (source.name, restored.name)
    src = np.asarray(Image.open(source))
    out = np.asarray(Image.open(restored))
    assert out.dtype == np.uint8
    assert out.shape[:2] == (src.shape[0] * 2, src.shape[1] * 2), (src.shape, out.shape)
    assert np.isfinite(out).all()
    assert int(out.min()) >= 0 and int(out.max()) <= 255
print(f"PASS: {len(outputs)} PNG outputs, valid x2 dimensions and [0,255] values")
PY
