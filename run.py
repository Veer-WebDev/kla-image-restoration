#!/usr/bin/env python3
"""KLA restoration evaluator entry point (.npy contract).

Usage
-----
    python run.py <input-dir> <output-dir>

Behaviour (per the KLA final submission check)
----------------------------------------------
* Reads every ``.npy`` file from ``<input-dir>`` (recursively).
* Creates ``<output-dir>`` if it does not exist.
* Writes one restored ``.npy`` per input, with the SAME filename.
* Each output is a grayscale array of shape ``(H, W)``.
* Output values are clamped to ``[0, 1]`` with no NaN or Inf.
* Restored images are produced at the target resolution (input size x scale,
  where scale comes from the bundled checkpoint; default 2).

The bundled weights live in ``models/`` and are loaded with no network access,
no API keys, and no user interaction. CUDA is used automatically when present,
otherwise CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kla_restore.checkpoint import load_model  # noqa: E402
from kla_restore.utils import select_device  # noqa: E402

# Checkpoint search order. ``models/`` is the submission-mandated location; the
# rest keep the script working inside the development tree without edits.
CHECKPOINT_CANDIDATES = (
    "models/best.pth",
    "models/model.pth",
    "models/final_model.pth",
    "weights/final_model.pth",
    "runs/config_acceptance/kla_restoration_submission_seed20260817/best.pth",
)


def find_checkpoint() -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        path = ROOT / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "no model weights found; expected one of: " + ", ".join(CHECKPOINT_CANDIDATES)
    )


def load_npy_gray(path: Path) -> np.ndarray:
    """Load a ``.npy`` array as float32 grayscale ``(H, W)`` without clipping.

    NoisyLR values may legitimately fall outside ``[0, 1]``; that range is
    preserved on input because the model was trained to exploit it. Multi-channel
    inputs are reduced to luminance so the output is single-channel.
    """
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            array = array.mean(axis=2)
    elif array.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D or 3D array, got shape {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return array


@torch.inference_mode()
def restore_one(model: torch.nn.Module, array: np.ndarray, scale: int, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(array)[None, None].to(device)  # (1, 1, H, W)
    restored = model(tensor, scale=scale, clamp=True)
    out = restored[0, 0].float().cpu().numpy()
    # Defensive: guarantee the mandated output invariants regardless of model.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python run.py <input-dir> <output-dir>", file=sys.stderr)
        return 2

    input_dir = Path(argv[0])
    output_dir = Path(argv[1])
    if not input_dir.is_dir():
        print(f"input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    inputs = sorted(p for p in input_dir.rglob("*.npy") if p.is_file())
    if not inputs:
        print(f"no .npy files found in {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device("auto")
    checkpoint = find_checkpoint()
    model, meta = load_model(checkpoint, map_location=device)
    model = model.to(device).eval()
    scale = int(meta.inference_scale or 2)

    print(f"checkpoint : {checkpoint}")
    print(f"device     : {device}")
    print(f"scale      : x{scale}")
    print(f"inputs     : {len(inputs)} .npy files from {input_dir}")

    started = time.perf_counter()
    failures = 0
    for index, path in enumerate(inputs, start=1):
        try:
            array = load_npy_gray(path)
            out = restore_one(model, array, scale, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Preserve filename relative to the input root.
            out_path = output_dir / path.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, out)
            print(f"[{index}/{len(inputs)}] {path.name} {array.shape} -> {out.shape}")
        except Exception as exc:  # keep going, report at the end
            failures += 1
            print(f"[{index}/{len(inputs)}] FAILED on {path.name}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    written = len(inputs) - failures
    print(f"restored {written}/{len(inputs)} images in {elapsed:.2f}s -> {output_dir}")
    if written == 0:
        return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
