"""Synthetic Drift-Sense dataset generator (DRAM- and FinFET-style SEM).

This module produces reference/wide-search grayscale image pairs for the
Applied Materials "Drift-Sense" localization task, following the official
sample prompt and calibration:

  * Both images are 1000x1000 px, grayscale.
  * Reference is a high-resolution ("100x") capture at 1 nm/px (1 um FOV).
  * Wide-search is a low-resolution ("10x") capture at 10 nm/px (10 um FOV),
    covering exactly 10x the physical area at the same pixel count.
  * We build a large continuous 10000x10000 px "fine canvas" at 1 nm/px, take
    a random 1000x1000 crop as the Reference, and downsample the whole canvas
    by 10x to make the Search. The Reference therefore appears shrunk by 10x
    somewhere inside the Search, exactly as specified.
  * Ground truth is the crop's center, expressed in Search-image pixels.

Structural styles (participant's choice, judged equally) are backed by public
descriptions of DRAM and FinFET layout; noise/imaging effects are backed by
public SEM-imaging literature. See docs/EXTERNAL_RESOURCES.md and the module
constants for citations. No proprietary fab data is used.

Noise / imaging model and sources
----------------------------------
1. Edge brightening ("edge effect"): secondary-electron yield rises at feature
   sidewalls, so edges image brighter than flat tops/troughs. Reimer,
   "Scanning Electron Microscopy" (Springer, 2nd ed.), ch. on SE contrast.
2. Poisson shot noise (dose dependent): SEM pixel intensity is a photon/electron
   count; noise variance scales with the mean. Standard sensor model, e.g.
   Janesick, "Photon Transfer" (SPIE, 2007).
3. Additive Gaussian detector/readout noise: independent per capture, applied
   separately to reference and search (they are two separate acquisitions).
   Standard EMCCD/PMT readout-noise model (Janesick, ibid.).
4. Optional multiplicative speckle: coherent-imaging granular noise, used to
   make the search image noisier for the robustness stress test the FAQ asks
   for. Goodman, "Speckle Phenomena in Optics" (2007).

Everything is deterministic given a seed.
"""

from __future__ import annotations

import cv2
import numpy as np

REFERENCE_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1
PIXEL_SIZE_SEARCH_NM = 10
SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM  # 10
FINE_CANVAS_SIZE_PX = REFERENCE_SIZE_PX * SCALE_FACTOR    # 10000

BACKGROUND = 40
LINE_VAL = 160
CONTACT_VAL = 225
GATE_VAL = 120

MANIFEST_COLUMNS = ("id", "architecture", "reference_path", "search_path",
                    "gt_x", "gt_y")


# --------------------------------------------------------------------------
# Structure rendering (fine canvas, 1 nm/px)
# --------------------------------------------------------------------------
def _line_positions(size_px: int, pitch_nm: float, jitter_nm: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Evenly-pitched line centers with small placement jitter (line-edge
    roughness / CD variation proxy)."""
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, jitter_nm)
    return np.asarray(positions)


def _stripe_mask(size_px: int, positions: np.ndarray, width_nm: float,
                 rng: np.random.Generator) -> np.ndarray:
    """1D boolean mask: True inside each line, with per-line width jitter."""
    mask = np.zeros(size_px, dtype=bool)
    widths = np.clip(width_nm * (1.0 + rng.normal(0, 0.1, size=len(positions))),
                     width_nm * 0.5, width_nm * 1.5)
    for center, w in zip(positions, widths):
        lo = max(int(round(center - w / 2.0)), 0)
        hi = min(int(round(center + w / 2.0)), size_px)
        mask[lo:hi] = True
    return mask


def generate_dram_canvas(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """DRAM-style: periodic horizontal word lines and vertical bit lines
    crossing at right angles, with a storage-node contact dot at a
    checkerboard of intersections (folded-bitline layout)."""
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    wl = _line_positions(size_px, pitch_nm=90, jitter_nm=1.5, rng=rng)
    bl = _line_positions(size_px, pitch_nm=90, jitter_nm=1.5, rng=rng)
    row = _stripe_mask(size_px, wl, width_nm=45, rng=rng)
    col = _stripe_mask(size_px, bl, width_nm=45, rng=rng)
    canvas[row, :] = np.maximum(canvas[row, :], LINE_VAL)
    canvas[:, col] = np.maximum(canvas[:, col], LINE_VAL + 10)
    for i, y in enumerate(wl):
        for j, x in enumerate(bl):
            if (i + j) % 2 == 0:
                r = max(1, int(round(12 * (1.0 + rng.normal(0, 0.1)))))
                cv2.circle(canvas, (int(round(x)), int(round(y))), r, CONTACT_VAL, -1)
    return canvas


def generate_finfet_canvas(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """FinFET-style: a dense set of parallel vertical fin lines, crossed by
    one or two horizontal gate bars."""
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    fins = _line_positions(size_px, pitch_nm=48, jitter_nm=1.0, rng=rng)
    fin_mask = _stripe_mask(size_px, fins, width_nm=24, rng=rng)
    canvas[:, fin_mask] = np.maximum(canvas[:, fin_mask], LINE_VAL)
    n_gates = int(rng.integers(1, 3))
    gate_pos = _line_positions(size_px, pitch_nm=size_px / (n_gates + 1),
                               jitter_nm=5.0, rng=rng)[:n_gates]
    gate_mask = _stripe_mask(size_px, gate_pos, width_nm=120, rng=rng)
    canvas[gate_mask, :] = np.maximum(canvas[gate_mask, :], GATE_VAL)
    return canvas


_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}


# --------------------------------------------------------------------------
# Large-scale zone composition (array "mats" separated by peripheral "strips")
# --------------------------------------------------------------------------
# A real die is not one endless periodic array. Memory arrays are broken into
# blocks ("mats") separated by peripheral/routing regions ("strips"). Those
# boundaries are what make localization tractable: a purely periodic canvas has
# no unique anchor, so every position looks identical (see the ambiguity-ceiling
# discussion in docs/submission). We therefore tile the fine canvas with mats of
# periodic pattern separated by strips of a different material.
STRIP_VAL = 90


def generate_zoned_canvas(size_px: int, architecture: str,
                          rng: np.random.Generator, *,
                          mat_size_nm: float = 2600.0,
                          strip_width_nm: float = 320.0) -> np.ndarray:
    """Compose the fine canvas from periodic-array mats separated by strips.

    The strips (peripheral material) provide the unique large-scale landmarks
    that let a matcher disambiguate one array block from an identical-looking
    one.
    """
    full = _GENERATORS[architecture](size_px, rng)
    sw = int(round(strip_width_nm))
    # Non-periodic strip placement: spacing is jittered so the local pattern of
    # gaps between strips is locally unique. A perfectly periodic strip grid
    # would itself be ambiguous (every strip looks like every other); real dies
    # break this with irregular block sizes and alignment features.
    for axis in (0, 1):
        pos = int(rng.integers(0, int(mat_size_nm)))
        while pos < size_px:
            if axis == 0:
                full[pos:min(pos + sw, size_px), :] = STRIP_VAL
            else:
                full[:, pos:min(pos + sw, size_px)] = STRIP_VAL
            step = mat_size_nm * rng.uniform(0.55, 1.6) + strip_width_nm
            pos += int(round(step))

    # Sparse alignment fiducials at random positions. Real wafer navigation
    # anchors on registration marks / distinctive features; a random
    # constellation of marks is locally unique (like star-field registration),
    # which is what lets a 1 um reference window be placed unambiguously even
    # inside an otherwise periodic array. Density ~1 mark per (400 nm)^2 so a
    # 1000 nm reference FOV typically contains several, giving a locally unique
    # constellation of positions and sizes.
    n_marks = int((size_px / 400.0) ** 2)
    for _ in range(n_marks):
        mx = int(rng.integers(0, size_px))
        my = int(rng.integers(0, size_px))
        r = int(rng.integers(14, 40))
        val = int(rng.integers(200, 256))
        cv2.rectangle(full, (mx - r, my - r), (mx + r, my + r), val, -1)
    return full


# --------------------------------------------------------------------------
# SEM imaging model (edge brightening, shot noise, readout noise, speckle)
# --------------------------------------------------------------------------
def _edge_brighten(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Brighten feature edges (SEM secondary-electron edge effect). We take the
    gradient magnitude and add it back, so sidewalls image brighter."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    return img.astype(np.float32) + strength * mag


def _charging_streaks(f: np.ndarray, rng: np.random.Generator, *,
                      prob_per_100_rows: float, intensity: float) -> np.ndarray:
    """SEM charging artefact: bright horizontal streaks where accumulated
    surface charge deflects the beam. Modelled as occasional partial rows with
    an additive brightness boost. (Reimer, SEM; charging contrast.)"""
    if prob_per_100_rows <= 0:
        return f
    h, w = f.shape
    n = rng.poisson(prob_per_100_rows * h / 100.0)
    for _ in range(int(n)):
        y = int(rng.integers(0, h))
        x0 = int(rng.integers(0, w))
        length = int(rng.integers(w // 8, w))
        boost = intensity * rng.uniform(20, 60)
        f[y, x0:x0 + length] += boost
    return f


def _barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Barrel(+)/pincushion(-) geometric distortion. Remaps pixels radially by
    (1 + k r^2); small k mimics SEM scan-field non-linearity."""
    if abs(k) < 1e-6:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.indices((h, w), dtype=np.float32)
    nx = (xs - cx) / cx
    ny = (ys - cy) / cy
    r2 = nx * nx + ny * ny
    factor = 1.0 + k * r2
    map_x = (cx + nx * factor * cx).astype(np.float32)
    map_y = (cy + ny * factor * cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _rotate_small(img: np.ndarray, deg: float) -> np.ndarray:
    """Small in-plane rotation (1-3 deg): stage/scan misalignment."""
    if abs(deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _scale_about_center(img: np.ndarray, scale: float) -> np.ndarray:
    """Apply a small isotropic feature-scale (CD/magnification) mismatch.

    The task slides ask for polygon scaling of -20% to +20%.  At the rendered
    image level, an affine scale about the scan-field center is the faithful
    equivalent: every drawn polygon becomes wider/narrower and its position is
    transformed consistently.  It is deliberately applied to the Search only
    so the localizer is tested against a cross-capture scale mismatch.  The
    caller transforms the ground-truth location by the same affine map.
    """
    if scale <= 0:
        raise ValueError("feature scale must be positive")
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _apply_sem_noise(img: np.ndarray, rng: np.random.Generator, *,
                     dose: float, readout_sigma: float, speckle_sigma: float,
                     edge_strength: float,
                     charging_prob: float = 0.0, charging_intensity: float = 0.0,
                     barrel_k: float = 0.0, rotation_deg: float = 0.0
                     ) -> np.ndarray:
    """Turn a clean structure image into a realistic SEM capture.

    dose            : higher = less shot noise (Poisson mean scales with dose)
    readout_sigma   : additive Gaussian detector/readout noise (independent
                      per capture)
    speckle_sigma   : multiplicative coherent noise (0 = off)
    edge_strength   : SE edge-brightening amount
    charging_*      : bright charging streaks (prob per 100 rows; intensity)
    barrel_k        : geometric barrel/pincushion distortion strength
    rotation_deg    : small whole-image rotation in degrees
    """
    f = _edge_brighten(img, edge_strength)
    f = np.clip(f, 0, 255)
    # Poisson shot noise: scale to a photon count set by dose, sample, scale back.
    scaled = f / 255.0 * dose
    shot = rng.poisson(np.clip(scaled, 0, None)).astype(np.float32)
    f = shot / max(dose, 1e-6) * 255.0
    if speckle_sigma > 0:
        f = f * (1.0 + rng.normal(0, speckle_sigma, size=f.shape))
    f = f + rng.normal(0, readout_sigma, size=f.shape)
    f = _charging_streaks(f, rng, prob_per_100_rows=charging_prob,
                          intensity=charging_intensity)
    f = np.clip(f, 0, 255).astype(np.uint8)
    # Geometric effects last, on the final grayscale capture.
    f = _barrel_distortion(f, barrel_k)
    f = _rotate_small(f, rotation_deg)
    return f


def image_reference(crop: np.ndarray, rng: np.random.Generator, *,
                    dose: float = 2000.0, readout_sigma: float = 2.0,
                    edge_strength: float = 0.6) -> np.ndarray:
    """High-dose, low-noise reference capture (1 nm/px, native crop)."""
    return _apply_sem_noise(crop, rng, dose=dose, readout_sigma=readout_sigma,
                            speckle_sigma=0.0, edge_strength=edge_strength)


def image_search(fine_canvas: np.ndarray, rng: np.random.Generator, *,
                 dose: float = 200.0, readout_sigma: float = 5.0,
                 speckle_sigma: float = 0.0, edge_strength: float = 0.6,
                 charging_prob: float = 0.0, charging_intensity: float = 0.0,
                 barrel_k: float = 0.0, rotation_deg: float = 0.0,
                 feature_scale: float = 1.0
                 ) -> np.ndarray:
    """Low-dose, noisier wide-search capture: downsample the fine canvas by
    10x (10 nm/px) then apply independent SEM noise."""
    small = cv2.resize(fine_canvas, (REFERENCE_SIZE_PX, REFERENCE_SIZE_PX),
                       interpolation=cv2.INTER_AREA)
    capture = _apply_sem_noise(small, rng, dose=dose,
                               readout_sigma=readout_sigma,
                               speckle_sigma=speckle_sigma,
                               edge_strength=edge_strength,
                               charging_prob=charging_prob,
                               charging_intensity=charging_intensity,
                               barrel_k=barrel_k, rotation_deg=rotation_deg)
    return _scale_about_center(capture, feature_scale)


# --------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------
def _to_optical_rgb(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bonus: turn a grayscale SEM-style capture into a plausible 3-channel
    optical-microscope image. Optical tools image the same structure in colour
    because different materials reflect different wavelengths; we approximate
    this with a smooth intensity-to-colour map (dark substrate -> blue-grey,
    mid metal -> warm gold, bright contacts/marks -> near-white) plus mild
    per-channel gain, so the pattern is preserved but rendered in colour. The
    localizer, which works on luminance, handles this transparently."""
    g = gray.astype(np.float32) / 255.0
    low = np.array([120, 90, 60], np.float32)     # BGR dark: blue-grey
    mid = np.array([70, 150, 200], np.float32)     # BGR mid: warm gold
    high = np.array([245, 245, 250], np.float32)   # BGR bright: near-white
    t = g[..., None]
    lo_mid = low + (mid - low) * np.clip(t / 0.5, 0, 1)
    mid_hi = mid + (high - mid) * np.clip((t - 0.5) / 0.5, 0, 1)
    rgb = np.where(t < 0.5, lo_mid, mid_hi)
    gain = 1.0 + rng.normal(0, 0.03, size=3).astype(np.float32)
    rgb = np.clip(rgb * gain, 0, 255)
    return rgb.astype(np.uint8)


def generate_sample(architecture: str, rng: np.random.Generator, *,
                    search_speckle_sigma: float = 0.0,
                    search_readout_sigma: float = 5.0,
                    zoned: bool = True, rgb: bool = False,
                    charging_prob: float = 0.0, charging_intensity: float = 0.0,
                    barrel_k: float = 0.0, rotation_max_deg: float = 0.0,
                    feature_scale_min: float = 1.0,
                    feature_scale_max: float = 1.0) -> dict:
    """Generate one reference/search pair with ground-truth center.

    Returns a dict with reference_img, search_img (both uint8 1000x1000),
    gt_x, gt_y (center of the reference FOV in search pixels), and architecture.

    ``zoned`` (default True) composes the canvas from array mats separated by
    peripheral strips so most crops contain a unique landmark. Set False for a
    purely periodic canvas (useful to demonstrate the ambiguity ceiling).

    Extra search-image augmentations (all default off) mirror the task slides:
    ``charging_prob``/``charging_intensity`` (charging streaks), ``barrel_k``
    (geometric distortion), ``rotation_max_deg`` (a small random rotation, e.g.
    1-3 deg), and ``feature_scale_min``/``feature_scale_max`` (the slide's
    global -20%..+20% polygon/CD scale sweep). These perturb the Search only;
    the Reference stays clean, which is realistic (reference is a careful
    high-dose capture). The returned ground truth is transformed to remain
    correct under feature scaling.
    """
    if architecture not in _GENERATORS:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"choose from {sorted(_GENERATORS)}")
    if feature_scale_min <= 0 or feature_scale_max <= 0:
        raise ValueError("feature scale bounds must be positive")
    if feature_scale_min > feature_scale_max:
        raise ValueError("feature_scale_min must be <= feature_scale_max")
    if zoned:
        fine = generate_zoned_canvas(FINE_CANVAS_SIZE_PX, architecture, rng)
    else:
        fine = _GENERATORS[architecture](FINE_CANVAS_SIZE_PX, rng)

    max_off = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    x0 = int(rng.integers(0, max_off + 1))
    y0 = int(rng.integers(0, max_off + 1))
    crop = fine[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = image_reference(crop, rng)
    rot = float(rng.uniform(-rotation_max_deg, rotation_max_deg)) if rotation_max_deg else 0.0
    feature_scale = float(rng.uniform(feature_scale_min, feature_scale_max))
    search_img = image_search(fine, rng, speckle_sigma=search_speckle_sigma,
                              readout_sigma=search_readout_sigma,
                              charging_prob=charging_prob,
                              charging_intensity=charging_intensity,
                              barrel_k=barrel_k, rotation_deg=rot,
                              feature_scale=feature_scale)

    if rgb:
        reference_img = _to_optical_rgb(reference_img, rng)
        search_img = _to_optical_rgb(search_img, rng)

    box = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100
    gt_x = x0 / SCALE_FACTOR + box / 2.0
    gt_y = y0 / SCALE_FACTOR + box / 2.0
    # Keep the label in the transformed Search coordinate system. The affine
    # scale is around the (500, 500) scan-field center.
    search_center = REFERENCE_SIZE_PX / 2.0
    gt_x = search_center + feature_scale * (gt_x - search_center)
    gt_y = search_center + feature_scale * (gt_y - search_center)
    return {"reference_img": reference_img, "search_img": search_img,
            "gt_x": gt_x, "gt_y": gt_y, "architecture": architecture,
            "feature_scale": feature_scale, "rotation_deg": rot}
