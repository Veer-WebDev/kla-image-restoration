#!/usr/bin/env bash
# End-to-end smoke: fixtures -> train -> inference -> assertions.
# Internal dev tool. Proves the mandated CLI interface works on this machine.
set -u

cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
# Repo-relative on purpose: MSYS rewrites /tmp for the .exe but not for Python's
# Path(), so an absolute /tmp path would point at two different directories.
OUT=runs/_smoke_restored
REPORT=runs/_smoke_restored_report.json

echo "=== ledger rows ==="
wc -l < results/experiments.csv

echo "=== inference (mandated interface) ==="
rm -rf "$OUT"
$PY inference.py \
  --input_dir data/fixtures/NoisyLR \
  --output_dir "$OUT" \
  --checkpoint runs/smoke_e2e/best.pth \
  --scale 2 \
  --report "$REPORT"
echo "INFER_EXIT=$?"

echo "=== outputs ==="
ls "$OUT" | head -5
echo "count: $(ls "$OUT" | wc -l)"

echo "=== size assertions ==="
$PY scripts/_assert_sizes.py "$OUT"
echo "ASSERT_EXIT=$?"
