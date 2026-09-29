"""
IBVAP Stage 2 — Degradation-Aware Visibility & Quality Assessment
===================================================================
Structured quality assessment module that identifies primary and secondary image
degradations (Low Light, High Brightness, Blur, Haze, Noise, Low Contrast).

Prevents incorrect generic low-light enhancement on bright or blurry frames.
Exposes configurable parameters:
- LOW_LIGHT_THRESHOLD, HIGH_BRIGHTNESS_THRESHOLD, CONTRAST_THRESHOLD, BLUR_THRESHOLD, HAZE_THRESHOLD, NOISE_THRESHOLD
"""

import cv2
import numpy as np

# ─── Configurable Quality & Threshold Constants ───────────────────────────────

LOW_LIGHT_THRESHOLD = 90.0        # Mean pixel intensity (0-255) below which frame is low light
HIGH_BRIGHTNESS_THRESHOLD = 190.0 # Mean pixel intensity above which frame is bright (do not apply gamma/CLAHE)
CONTRAST_THRESHOLD = 35.0         # Standard deviation below which contrast is low
BLUR_THRESHOLD = 60.0             # Laplacian variance below which frame is blurry
HAZE_THRESHOLD = 0.40             # Dark Channel Prior ratio above which frame is hazy
NOISE_THRESHOLD = 12.0            # High-frequency Laplacian noise estimate threshold
MAX_CLIPPING_PCT = 5.0            # Maximum acceptable highlight clipping percentage (% pixels > 250)


def compute_dark_channel(frame, patch_size=15):
    """Compute dark channel prior metric for haze detection."""
    if frame is None or frame.size == 0:
        return 0.0
    min_channel = np.min(frame, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return float(np.mean(dark_channel)) / 255.0


def compute_noise_estimate(gray):
    """Estimate high-frequency Gaussian noise level using median absolute deviation of Laplacian."""
    if gray is None or gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = float(np.median(np.abs(lap)) * 1.4826)
    return round(sigma, 1)


def compute_highlight_clipping(gray):
    """Calculate percentage of overexposed pixels near white (>= 250)."""
    if gray is None or gray.size == 0:
        return 0.0
    clipped = np.sum(gray >= 250)
    total = gray.size
    return round(float(clipped / total) * 100.0, 2)


def assess_visibility(frame):
    """
    Perform structured degradation classification.

    Returns dict containing:
    - state: "GOOD" or "POOR"
    - primary_degradation: e.g. "LOW_LIGHT", "HIGH_BRIGHTNESS", "BLURRY", "HAZY", "NOISY", "LOW_CONTRAST", "GOOD"
    - secondary_degradations: list of additional issues
    - action_summary: recommended enhancement action
    - numerical metrics: brightness, contrast, sharpness, haze_level, noise_level, highlight_clipping
    """
    if frame is None or frame.size == 0:
        return {
            "state": "GOOD",
            "primary_degradation": "GOOD",
            "secondary_degradations": [],
            "action_summary": "NO ENHANCEMENT (RAW)",
            "is_poor": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "haze_level": 0.0,
            "noise_level": 0.0,
            "highlight_clipping": 0.0,
            "reasons": []
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    haze_level = compute_dark_channel(frame)
    noise_level = compute_noise_estimate(gray)
    highlight_clipping = compute_highlight_clipping(gray)

    reasons = []

    # Check degradation conditions
    if brightness < LOW_LIGHT_THRESHOLD:
        reasons.append("LOW_LIGHT")
    if brightness > HIGH_BRIGHTNESS_THRESHOLD:
        reasons.append("HIGH_BRIGHTNESS")
    if contrast < CONTRAST_THRESHOLD:
        reasons.append("LOW_CONTRAST")
    if sharpness < BLUR_THRESHOLD:
        reasons.append("BLURRY")
    if haze_level > HAZE_THRESHOLD and contrast < 45.0:
        reasons.append("HAZY")
    if noise_level > NOISE_THRESHOLD:
        reasons.append("NOISY")

    # Determine Primary Degradation
    if "LOW_LIGHT" in reasons:
        primary = "LOW_LIGHT"
    elif "HIGH_BRIGHTNESS" in reasons and "BLURRY" in reasons:
        primary = "BLURRY"
    elif "HIGH_BRIGHTNESS" in reasons:
        primary = "HIGH_BRIGHTNESS"
    elif "HAZY" in reasons:
        primary = "HAZY"
    elif "BLURRY" in reasons:
        primary = "BLURRY"
    elif "NOISY" in reasons:
        primary = "NOISY"
    elif "LOW_CONTRAST" in reasons:
        primary = "LOW_CONTRAST"
    else:
        primary = "GOOD"

    secondary = [r for r in reasons if r != primary]
    is_poor = (primary != "GOOD")

    # Determine Recommended Action Summary
    if primary in ["HIGH_BRIGHTNESS", "BLURRY"]:
        action = "NO AGGRESSIVE ENHANCEMENT"
    elif primary == "LOW_LIGHT":
        action = "GAMMA + CLAHE"
    elif primary == "HAZY":
        action = "DEHAZE"
    elif primary == "NOISY":
        action = "DENOISE"
    elif primary == "LOW_CONTRAST":
        action = "MILD CLAHE"
    else:
        action = "NO ENHANCEMENT (RAW)"

    return {
        "state": "POOR" if is_poor else "GOOD",
        "primary_degradation": primary,
        "secondary_degradations": secondary,
        "action_summary": action,
        "is_poor": is_poor,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "haze_level": round(haze_level, 3),
        "noise_level": round(noise_level, 1),
        "highlight_clipping": highlight_clipping,
        "reasons": reasons,
    }
