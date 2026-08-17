#!/usr/bin/env python3
"""Inference entry point mandated by the KLA problem statement.

Contract
--------
    python inference.py --input_dir <NoisyLR dir> --output_dir <restored dir>

Every image in ``--input_dir`` produces one restored image of the same base name in
``--output_dir``. The evaluator runs this file unmodified.

Dependencies: torch, numpy, Pillow. Nothing else. No LPIPS, no scikit-image, no
network access, no pandas. If CUDA is unavailable it runs on CPU automatically.

Output size
-----------
GT is not available at inference time, so the output size cannot be read from it.
Resolution order (assumption A1 in docs/IMPLEMENTATION_AUDIT.md):

1. ``--target-size H W``            explicit, applied to every image;
2. ``--size-map FILE``             per-file ``name,H,W`` or JSON mapping;
3. ``--scale N``                   explicit factor;
4. the ``inference_scale`` recorded in the checkpoint;
5. ``2`` as the final fallback.

The resolved rule is printed for every run so the choice is never silent.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kla_restore.checkpoint import load_model  # noqa: E402
from kla_restore.utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    get_logger,
    image_files,
    load_image_float,
    save_image_float,
    select_device,
    setup_logging,
    to_numpy,
    to_tensor,
)

LOGGER = get_logger()

DEFAULT_CHECKPOINT_CANDIDATES = (
    "checkpoints/best.pth",
    "checkpoints/model.pth",
    "runs/baseline_residual_unet/best.pth",
    "best.pth",
    "model.pth",
)


def find_checkpoint(explicit: str | None) -> Path:
    """Locate the weights, searching the conventional locations when not given."""
    root = Path(__file__).resolve().parent
    if explicit:
        path = Path(explicit)
        if not path.is_absolute() and not path.exists():
            path = root / explicit
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {explicit}")
        return path
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path = root / candidate
        if path.exists():
            LOGGER.info("using checkpoint %s", path)
            return path
    raise FileNotFoundError(
        "no checkpoint found. Pass --checkpoint PATH, or place the weights at "
        f"one of: {', '.join(DEFAULT_CHECKPOINT_CANDIDATES)}"
    )


def load_size_map(path: str | Path) -> dict[str, tuple[int, int]]:
    """Load per-image target sizes from CSV (``name,H,W``) or JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"size map not found: {path}")
    mapping: dict[str, tuple[int, int]] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, dict):
                mapping[Path(key).stem] = (int(value["height"]), int(value["width"]))
            else:
                h, w = value
                mapping[Path(key).stem] = (int(h), int(w))
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 3:
                raise ValueError(f"size map rows need name,H,W; got {row}")
            if not row[1].strip().isdigit():  # tolerate a header line
                continue
            mapping[Path(row[0].strip()).stem] = (int(row[1]), int(row[2]))
    return mapping


def resolve_target_size(
    input_hw: tuple[int, int],
    stem: str,
    *,
    target_size: tuple[int, int] | None,
    size_map: dict[str, tuple[int, int]] | None,
    scale: int,
) -> tuple[tuple[int, int], str]:
    """Apply the documented resolution order. Returns ``(size, reason)``."""
    if target_size is not None:
        return target_size, "--target-size"
    if size_map and stem in size_map:
        return size_map[stem], "--size-map"
    return (input_hw[0] * scale, input_hw[1] * scale), f"scale x{scale}"


@torch.inference_mode()
def restore_tensor(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    target_size: tuple[int, int],
    *,
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> torch.Tensor:
    """Restore one ``(1, C, h, w)`` tensor, optionally tiled for large inputs.

    Tiling exists so a 4096-pixel image cannot exhaust memory on the evaluator's
    machine. Tiles overlap and are blended with a linear ramp, so seams do not
    appear in the output.
    """
    if tile_size <= 0 or (noisy.shape[-2] <= tile_size and noisy.shape[-1] <= tile_size):
        return model(noisy, target_size=target_size, clamp=True)

    _, channels, height, width = noisy.shape
    out_h, out_w = target_size
    scale_h = out_h / height
    scale_w = out_w / width
    output = torch.zeros((1, channels, out_h, out_w), dtype=torch.float32, device=noisy.device)
    weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=noisy.device)
    step = max(1, tile_size - tile_overlap)

    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom = min(height, top + tile_size)
            right = min(width, left + tile_size)
            top_a = max(0, bottom - tile_size)
            left_a = max(0, right - tile_size)
            tile = noisy[:, :, top_a:bottom, left_a:right]
            tile_out_h = max(1, int(round((bottom - top_a) * scale_h)))
            tile_out_w = max(1, int(round((right - left_a) * scale_w)))
            restored = model(tile, target_size=(tile_out_h, tile_out_w), clamp=True)

            oy = min(int(round(top_a * scale_h)), max(0, out_h - tile_out_h))
            ox = min(int(round(left_a * scale_w)), max(0, out_w - tile_out_w))
            ramp = _blend_window(tile_out_h, tile_out_w, restored.device)
            output[:, :, oy : oy + tile_out_h, ox : ox + tile_out_w] += restored * ramp
            weight[:, :, oy : oy + tile_out_h, ox : ox + tile_out_w] += ramp

    return (output / weight.clamp_min(1e-8)).clamp(0.0, 1.0)


def _blend_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Separable linear ramp used to blend overlapping tiles."""
    def ramp(length: int) -> torch.Tensor:
        if length < 4:
            return torch.ones(length, device=device)
        edge = max(1, length // 8)
        values = torch.ones(length, device=device)
        taper = torch.linspace(0.1, 1.0, edge, device=device)
        values[:edge] = taper
        values[-edge:] = taper.flip(0)
        return values

    return (ramp(height)[:, None] * ramp(width)[None, :])[None, None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description="Restore NoisyLR semiconductor images to ground-truth resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # The two argument names below are fixed by the problem statement.
    parser.add_argument("--input_dir", required=True, help="directory of NoisyLR images")
    parser.add_argument("--output_dir", required=True, help="directory for restored images")
    parser.add_argument("--checkpoint", default=None, help="model weights (.pth)")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--scale", type=int, default=None, help="output = input size x scale")
    parser.add_argument(
        "--target-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="force every output to this size",
    )
    parser.add_argument("--size-map", default=None, help="CSV (name,H,W) or JSON of per-image sizes")
    parser.add_argument("--tile-size", type=int, default=0, help="tile large inputs; 0 disables")
    parser.add_argument("--tile-overlap", type=int, default=32, help="tile overlap in pixels")
    parser.add_argument("--bit-depth", type=int, default=None, choices=(8, 16), help="output bit depth")
    parser.add_argument("--out-ext", default=".png", help="output file extension")
    parser.add_argument("--suffix", default="", help="string appended to each output stem")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--log-file", default=None, help="also write logs to this file")
    parser.add_argument("--report", default=None, help="write a JSON run report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_file, level=args.log_level)
    started = time.perf_counter()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        LOGGER.error("input_dir does not exist: %s", input_dir)
        return 2
    if not input_dir.is_dir():
        LOGGER.error("input_dir is not a directory: %s", input_dir)
        return 2

    try:
        paths = image_files(input_dir)
    except Exception as exc:
        LOGGER.error("could not list %s: %s", input_dir, exc)
        return 2
    if not paths:
        LOGGER.error(
            "no images found in %s (looked for %s)", input_dir, ", ".join(sorted(IMAGE_EXTENSIONS))
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_path = find_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2

    device = select_device(args.device)
    try:
        model, meta = load_model(checkpoint_path, map_location=device)
    except Exception as exc:
        LOGGER.error("failed to load %s: %s", checkpoint_path, exc)
        return 2
    model = model.to(device).eval()

    scale = int(args.scale if args.scale is not None else (meta.inference_scale or 2))
    if scale < 1:
        LOGGER.error("scale must be >= 1, got %d", scale)
        return 2
    bit_depth = int(args.bit_depth if args.bit_depth is not None else (meta.bit_depth or 8))
    channels = int(meta.channels or model.config.in_channels)
    target_size = tuple(args.target_size) if args.target_size else None
    size_map = None
    if args.size_map:
        try:
            size_map = load_size_map(args.size_map)
        except Exception as exc:
            LOGGER.error("could not read size map: %s", exc)
            return 2

    LOGGER.info("=" * 78)
    LOGGER.info("checkpoint  : %s (experiment %s, epoch %d)", checkpoint_path, meta.experiment_id, meta.epoch)
    LOGGER.info("device      : %s", device)
    LOGGER.info("images      : %d from %s", len(paths), input_dir)
    LOGGER.info("channels    : %d", channels)
    LOGGER.info("output size : %s", "explicit" if target_size else (f"size map + x{scale}" if size_map else f"input x{scale}"))
    LOGGER.info("bit depth   : %d", bit_depth)
    LOGGER.info("=" * 78)

    records: list[dict[str, Any]] = []
    failures = 0
    for index, path in enumerate(paths, start=1):
        try:
            # clip=False: NoisyLR legitimately exceeds [0, 1] and clipping would
            # destroy information the model was trained to exploit.
            loaded = load_image_float(path, clip=False, grayscale=channels == 1)
            array = loaded.array
            if channels == 3 and array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            elif channels == 1 and array.shape[2] == 3:
                loaded = load_image_float(path, clip=False, grayscale=True)
                array = loaded.array

            size, reason = resolve_target_size(
                (array.shape[0], array.shape[1]),
                path.stem,
                target_size=target_size,
                size_map=size_map,
                scale=scale,
            )
            tensor = to_tensor(array)[None].to(device)
            image_started = time.perf_counter()
            restored = restore_tensor(
                model,
                tensor,
                size,
                tile_size=int(args.tile_size),
                tile_overlap=int(args.tile_overlap),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - image_started) * 1000.0

            out_path = output_dir / f"{path.stem}{args.suffix}{args.out_ext}"
            if out_path.exists() and not args.overwrite:
                LOGGER.warning("output exists, overwriting: %s", out_path.name)
            save_image_float(out_path, to_numpy(restored[0].float().cpu()), bit_depth=bit_depth)

            records.append(
                {
                    "input": path.name,
                    "output": out_path.name,
                    "input_size": [int(array.shape[0]), int(array.shape[1])],
                    "output_size": [int(size[0]), int(size[1])],
                    "size_reason": reason,
                    "ms": round(elapsed_ms, 2),
                }
            )
            LOGGER.info(
                "[%d/%d] %s %dx%d -> %dx%d (%s) %.1f ms",
                index,
                len(paths),
                path.name,
                array.shape[0],
                array.shape[1],
                size[0],
                size[1],
                reason,
                elapsed_ms,
            )
        except Exception as exc:
            failures += 1
            LOGGER.error("failed on %s: %s", path.name, exc, exc_info=args.log_level == "DEBUG")

    elapsed = time.perf_counter() - started
    per_image = [r["ms"] for r in records]
    LOGGER.info("=" * 78)
    LOGGER.info(
        "restored %d/%d images in %.2fs (mean %.1f ms, median %.1f ms)",
        len(records),
        len(paths),
        elapsed,
        float(np.mean(per_image)) if per_image else float("nan"),
        float(np.median(per_image)) if per_image else float("nan"),
    )
    if failures:
        LOGGER.warning("%d image(s) failed", failures)
    LOGGER.info("outputs in %s", output_dir.resolve())
    LOGGER.info("=" * 78)

    if args.report:
        report = {
            "checkpoint": str(checkpoint_path),
            "experiment_id": meta.experiment_id,
            "device": str(device),
            "torch_version": torch.__version__,
            "scale": scale,
            "bit_depth": bit_depth,
            "channels": channels,
            "n_inputs": len(paths),
            "n_outputs": len(records),
            "failures": failures,
            "total_seconds": round(elapsed, 3),
            "mean_ms": round(float(np.mean(per_image)), 3) if per_image else None,
            "median_ms": round(float(np.median(per_image)), 3) if per_image else None,
            "images": records,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("report written to %s", report_path)

    if not records:
        return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
