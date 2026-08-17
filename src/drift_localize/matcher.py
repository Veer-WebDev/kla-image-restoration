"""Normalized cross-correlation localizer for Drift-Sense.

The Reference is captured at 10x finer pixel size than the Search image, so a
1000x1000 Reference field of view occupies roughly a 100x100 window in the
Search image. We downsample the Reference to a set of candidate template sizes
(we do not hard-code exactly 10x, since a real solver is not handed the pixel
ratio), slide each template over the Search image with normalized
cross-correlation, and take the best-scoring location. A parabolic fit around
the winning correlation peak gives a sub-pixel centre.

An optional fine-verification stage re-scores the top-K correlation peaks using
the full-resolution Reference (upsampling the matched Search window back toward
the Reference scale). This is designed to break ties between the many
near-identical positions that periodic layouts (DRAM arrays) produce. On a
large held-out synthetic set it neither helps nor hurts the aggregate metric,
because the remaining errors are genuine appearance ambiguities (see
docs/submission), so it is off by default and offered for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_SCALES = (9.0, 9.5, 10.0, 10.5, 11.0)


@dataclass
class LocalizeResult:
    x: float
    y: float
    score: float          # correlation score of the winning peak
    scale: float          # template downscale factor that won
    n_tied_peaks: int     # candidate peaks within `tie_margin` of the best
    ambiguous: bool       # True when more than one strong peak competes


def _parabolic_subpixel(response: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Refine an integer correlation-peak location to sub-pixel using a 1-D
    parabola fit along each axis. Returns offsets in [-1, 1]."""
    h, w = response.shape
    dx = dy = 0.0
    if 0 < x < w - 1:
        left, center, right = response[y, x - 1], response[y, x], response[y, x + 1]
        denom = left - 2 * center + right
        if abs(denom) > 1e-12:
            dx = 0.5 * (left - right) / denom
    if 0 < y < h - 1:
        up, center, down = response[y - 1, x], response[y, x], response[y + 1, x]
        denom = up - 2 * center + down
        if abs(denom) > 1e-12:
            dy = 0.5 * (up - down) / denom
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))


def _iter_peaks(response: np.ndarray, k: int, min_dist: int):
    """Yield up to k local maxima of `response`, suppressing a square window of
    half-width `min_dist` around each returned peak so array aliases are seen as
    distinct candidates rather than one blurred blob."""
    work = response.copy()
    for _ in range(k):
        _, score, _, loc = cv2.minMaxLoc(work)
        if not np.isfinite(score):
            break
        x, y = loc
        yield score, x, y
        x0, x1 = max(0, x - min_dist), min(work.shape[1], x + min_dist + 1)
        y0, y1 = max(0, y - min_dist), min(work.shape[0], y + min_dist + 1)
        work[y0:y1, x0:x1] = -np.inf


def _fine_score(reference_f: np.ndarray, search: np.ndarray,
                x: int, y: int, tw: int, th: int) -> float:
    """NCC of the full-resolution Reference against the Search window at
    (x, y, tw, th), upsampled back to the Reference size."""
    Rh, Rw = reference_f.shape
    xi = min(max(x, 0), search.shape[1] - tw)
    yi = min(max(y, 0), search.shape[0] - th)
    window = search[yi:yi + th, xi:xi + tw].astype(np.float32)
    window_up = cv2.resize(window, (Rw, Rh), interpolation=cv2.INTER_CUBIC)
    a = window_up - window_up.mean()
    b = reference_f - reference_f.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-9
    return float((a * b).sum() / denom)


def predict(reference_path: str, search_path: str, *,
            scales=DEFAULT_SCALES, topk: int = 5, verify: bool = False,
            tie_margin: float = 0.03,
            center_tiebreak: bool = True) -> LocalizeResult:
    """Localize the Reference field of view inside the Search image.

    Parameters
    ----------
    reference_path, search_path : str
        Grayscale image paths.
    scales : iterable of float
        Candidate Reference-to-Search downscale factors to try.
    topk : int
        Correlation peaks retained per scale for tie counting / verification.
    verify : bool
        If True, re-rank the top peaks with the fine-resolution NCC stage.
    tie_margin : float
        Score gap under which a competing peak counts as a tie (ambiguity).
    center_tiebreak : bool
        When several peaks tie, follow the official Drift-Sense rule and
        report the tile whose center is closest to the search image center.
        Default True (spec-compliant). Set False to always take the highest
        correlation peak; on our crop-labeled synthetic GT that scores higher,
        but it does not follow the stated scoring convention.
    """
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise ValueError(f"Could not read {reference_path!r} or {search_path!r}")
    reference_f = reference.astype(np.float32)
    Rh, Rw = reference.shape

    candidates = []  # (coarse_score, x, y, dx, dy, tw, th, scale)
    for scale in scales:
        tw = max(int(round(Rw / scale)), 1)
        th = max(int(round(Rh / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        for score, x, y in _iter_peaks(response, topk, max(tw, th) // 2):
            dx, dy = _parabolic_subpixel(response, x, y)
            candidates.append((float(score), x, y, dx, dy, tw, th, scale))

    if not candidates:
        return LocalizeResult(search.shape[1] / 2.0, search.shape[0] / 2.0,
                              0.0, float("nan"), 0, True)

    if verify:
        # (rank_score, coarse_score, x, y, dx, dy, tw, th, scale)
        ranked = [
            (_fine_score(reference_f, search, x, y, tw, th) + 0.1 * s,
             s, x, y, dx, dy, tw, th, scale)
            for (s, x, y, dx, dy, tw, th, scale) in candidates
        ]
    else:
        ranked = [(s, s, x, y, dx, dy, tw, th, scale)
                  for (s, x, y, dx, dy, tw, th, scale) in candidates]
    ranked.sort(reverse=True)
    ranking = np.array([r[0] for r in ranked])

    best = ranking[0]
    # Peaks whose rank-score is within tie_margin of the best are "tied".
    tied = [r for r in ranked if r[0] >= best - tie_margin]
    n_tied = len(tied)

    # Spec: when more than one region matches, report the tile whose center
    # is closest to the search image's center; otherwise the single best peak.
    sh, sw = search.shape
    scx, scy = sw / 2.0, sh / 2.0

    def _center(r):
        _, _, x, y, dx, dy, tw, th, _ = r
        return (x + dx + tw / 2.0, y + dy + th / 2.0)

    if n_tied > 1 and center_tiebreak:
        winner = min(tied, key=lambda r: (lambda c: (c[0] - scx) ** 2 + (c[1] - scy) ** 2)(_center(r)))
    else:
        winner = ranked[0]

    _, score, x, y, dx, dy, tw, th, scale = winner
    cx = x + dx + tw / 2.0
    cy = y + dy + th / 2.0
    return LocalizeResult(cx, cy, float(score), float(scale), n_tied, n_tied > 1)
