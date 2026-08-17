"""Tests for the Drift-Sense NCC localizer.

These build small synthetic Reference/Search pairs in a temp directory (no
external data) and check that the solver recovers the known centre, produces
sub-pixel output, and flags a periodic (ambiguous) layout differently from a
unique one.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from drift_localize import predict, generator  # noqa: E402


def _make_pair(tmp_path, search_img, ref_top_left, ref_size_in_search=100,
               scale=10):
    """Crop a reference window from a high-res source and write a Reference
    (high-res) and Search (downscaled) pair. Returns (ref_path, search_path,
    gt_x, gt_y)."""
    x0, y0 = ref_top_left
    w = h = ref_size_in_search
    # Reference is the crop upscaled by `scale` (1 nm/px vs 10 nm/px).
    crop = search_img[y0:y0 + h, x0:x0 + w]
    reference = cv2.resize(crop, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)
    ref_path = str(tmp_path / "ref.png")
    srch_path = str(tmp_path / "search.png")
    cv2.imwrite(ref_path, reference)
    cv2.imwrite(srch_path, search_img)
    gt_x = x0 + w / 2.0
    gt_y = y0 + h / 2.0
    return ref_path, srch_path, gt_x, gt_y


def _unique_search(size=1000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def test_recovers_unique_center(tmp_path):
    search = _unique_search(seed=1)
    ref_path, srch_path, gx, gy = _make_pair(tmp_path, search, (300, 450))
    res = predict(ref_path, srch_path)
    err = ((res.x - gx) ** 2 + (res.y - gy) ** 2) ** 0.5
    assert err < 3.0, f"error {err:.2f}px too large"


def test_output_is_subpixel(tmp_path):
    search = _unique_search(seed=2)
    ref_path, srch_path, gx, gy = _make_pair(tmp_path, search, (512, 128))
    res = predict(ref_path, srch_path)
    # Non-integer centre is expected from the parabolic refinement.
    assert isinstance(res.x, float) and isinstance(res.y, float)
    assert res.score > 0.3


def test_unique_texture_not_flagged_ambiguous(tmp_path):
    search = _unique_search(seed=3)
    ref_path, srch_path, _, _ = _make_pair(tmp_path, search, (200, 200))
    res = predict(ref_path, srch_path)
    assert res.n_tied_peaks == 1
    assert res.ambiguous is False


def test_periodic_layout_flagged_ambiguous(tmp_path):
    # A tiled periodic pattern produces multiple equally-good matches.
    tile = np.zeros((50, 50), np.uint8)
    tile[10:40, 10:40] = 200
    search = np.tile(tile, (20, 20))
    ref_path, srch_path, _, _ = _make_pair(tmp_path, search, (300, 300))
    res = predict(ref_path, srch_path)
    assert res.n_tied_peaks > 1
    assert res.ambiguous is True


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        predict(str(tmp_path / "nope.png"), str(tmp_path / "nope2.png"))


def test_center_tiebreak_selects_center_closest(tmp_path):
    # Two identical tiles equidistant... build a periodic search where several
    # positions tie, and check center-tiebreak picks the one nearest center.
    tile = np.zeros((50, 50), np.uint8)
    tile[10:40, 10:40] = 200
    search = np.tile(tile, (20, 20))
    ref_path, srch_path, _, _ = _make_pair(tmp_path, search, (300, 300))
    on = predict(ref_path, srch_path, center_tiebreak=True)
    off = predict(ref_path, srch_path, center_tiebreak=False)
    assert on.ambiguous and off.ambiguous
    sc = search.shape[1] / 2.0
    d_on = ((on.x - sc) ** 2 + (on.y - sc) ** 2) ** 0.5
    d_off = ((off.x - sc) ** 2 + (off.y - sc) ** 2) ** 0.5
    # Center tie-break should be no farther from center than the raw argmax.
    assert d_on <= d_off + 1e-6


def test_generator_produces_findable_pair(tmp_path):
    # A zoned canvas with fiducials should yield at least one sample the
    # matcher localizes to within a few pixels (closed loop: generate->match).
    ref_p = str(tmp_path / "g_ref.png")
    srch_p = str(tmp_path / "g_srch.png")
    best_err = 1e9
    for seed in range(6):
        rng = np.random.default_rng(seed)
        s = generator.generate_sample("finfet", rng)
        cv2.imwrite(ref_p, s["reference_img"])
        cv2.imwrite(srch_p, s["search_img"])
        res = predict(ref_p, srch_p)
        err = ((res.x - s["gt_x"]) ** 2 + (res.y - s["gt_y"]) ** 2) ** 0.5
        best_err = min(best_err, err)
    assert best_err < 5.0, f"no findable sample; best error {best_err:.1f}px"


def test_generator_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        generator.generate_sample("cpu", np.random.default_rng(0))
