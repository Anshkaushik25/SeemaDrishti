"""
IBVAP Stage 2 — Degradation-Aware Enhancement Pipeline
========================================================
Adaptive enhancement pipeline that executes ONLY appropriate operations based on
specific degradation classification (Low Light, Low Contrast, Haze, Noise).

Exposes Highlight Protection & Rejection logic to prevent white clipping,
over-brightening, or visual degradation on bright/blurry images.
"""

import cv2
import numpy as np
from visibility_utils import (
    assess_visibility,
    compute_highlight_clipping,
    LOW_LIGHT_THRESHOLD,
    HIGH_BRIGHTNESS_THRESHOLD,
    CONTRAST_THRESHOLD,
    BLUR_THRESHOLD,
    HAZE_THRESHOLD,
    NOISE_THRESHOLD,
    MAX_CLIPPING_PCT,
)

# ─── Conservative Enhancement Parameters ─────────────────────────────────────

GAMMA = 1.3                  # Conservative Gamma boost for low-light recovery (range 1.2 - 1.5)
CLAHE_CLIP_LIMIT = 2.0       # Conservative CLAHE clip limit (prevents artificial contrast & halos)
CLAHE_TILE_GRID = (8, 8)     # Tile grid dimensions for local histogram equalization
DENOISE_STRENGTH = 20        # Bilateral filter sigma for edge-preserving denoising
ENABLE_DEHAZING = True       # Dehazing toggle


def apply_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID):
    """Apply CLAHE in LAB color space to enhance L channel without distorting color tint."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_gamma(frame, gamma=GAMMA):
    """Apply conservative Gamma correction for low-light luminance boost."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_denoising(frame, sigma=DENOISE_STRENGTH):
    """Apply lightweight edge-preserving bilateral filter."""
    return cv2.bilateralFilter(frame, d=5, sigmaColor=sigma, sigmaSpace=sigma)


def apply_dehazing(frame, haze_level):
    """Lightweight contrast recovery for hazy/foggy frames."""
    alpha = 1.0 + min(0.4, haze_level * 0.6)
    beta = -8.0 * haze_level
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def check_highlight_protection(raw_frame, enhanced_frame):
    """
    Highlight Protection Safety Check.

    Rejects enhancement if:
    1. Highlight clipping (% pixels >= 250) increases by more than 5.0%
    2. Enhanced mean brightness exceeds 235.0
    """
    raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

    raw_clip = compute_highlight_clipping(raw_gray)
    enh_clip = compute_highlight_clipping(enh_gray)

    raw_mean = float(np.mean(raw_gray))
    enh_mean = float(np.mean(enh_gray))

    # Rejection conditions
    if (enh_clip - raw_clip) > MAX_CLIPPING_PCT:
        return False, f"REJECTED (Highlight clipping increased from {raw_clip:.1f}% to {enh_clip:.1f}%)"

    if enh_mean > 235.0 and raw_mean <= 225.0:
        return False, f"REJECTED (Excessive brightness {enh_mean:.1f})"

    return True, "PASSED"


def enhance_conditionally(raw_frame, metrics=None):
    """
    Degradation-Aware Adaptive Enhancement Pipeline.

    Executes only the appropriate enhancement operations:
    - HIGH_BRIGHTNESS or BLURRY -> Returns raw_frame (avoids brightening/distortion).
    - LOW_LIGHT -> Applies conservative Gamma + CLAHE.
    - HAZY -> Applies Dehazing only.
    - NOISY -> Applies Denoising only.
    - LOW_CONTRAST -> Applies mild CLAHE.

    Enforces Highlight Protection Safety Check.
    RAW FRAME SAFETY: Never mutates raw_frame array.
    """
    if raw_frame is None or raw_frame.size == 0:
        return raw_frame

    if metrics is None:
        metrics = assess_visibility(raw_frame)

    raw_copy = raw_frame.copy()
    primary = metrics.get("primary_degradation", "GOOD")
    reasons = metrics.get("reasons", [])
    brightness = metrics.get("brightness", 100.0)

    # 1. High Brightness or Blurry Frames: Avoid Gamma / CLAHE
    if primary in ["HIGH_BRIGHTNESS", "BLURRY"] or brightness > HIGH_BRIGHTNESS_THRESHOLD:
        return raw_copy

    # 2. Good Frames: Return raw copy
    if primary == "GOOD" or not metrics.get("is_poor", False):
        return raw_copy

    enhanced = raw_copy.copy()

    # 3. Low Light Processing (Conservative Gamma + CLAHE)
    if primary == "LOW_LIGHT" or "LOW_LIGHT" in reasons:
        enhanced = apply_gamma(enhanced, gamma=GAMMA)
        enhanced = apply_clahe(enhanced, clip_limit=CLAHE_CLIP_LIMIT)

    # 4. Low Contrast Processing (Mild CLAHE)
    elif primary == "LOW_CONTRAST" or "LOW_CONTRAST" in reasons:
        enhanced = apply_clahe(enhanced, clip_limit=1.5)

    # 5. Dehazing Processing
    if primary == "HAZY" or "HAZY" in reasons:
        if ENABLE_DEHAZING:
            enhanced = apply_dehazing(enhanced, metrics.get("haze_level", 0.4))

    # 6. Denoising Processing
    if primary == "NOISY" or "NOISY" in reasons:
        enhanced = apply_denoising(enhanced, sigma=DENOISE_STRENGTH)

    # 7. Highlight Protection Safety Verification
    is_safe, reason = check_highlight_protection(raw_copy, enhanced)
    if not is_safe:
        return raw_copy  # Fallback to RAW if enhancement created overexposure or clipping

    return enhanced
