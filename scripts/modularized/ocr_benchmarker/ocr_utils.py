"""
ocr_utils.py
============
Shared preprocessing pipelines and filtering utilities used across all OCR processors.
Each preprocess_* function takes a BGR numpy array and returns a BGR numpy array
(white text on black background, 2× upscaled, with 20 px black border padding).
"""

import cv2
import numpy as np
import base64


# ── Preprocessing Pipelines ───────────────────────────────────────────────────

def preprocess_row_density(crop: np.ndarray) -> np.ndarray:
    """
    'Zelda Classic' — Row-density furigana suppression.

    Pipeline:
      1. Threshold at 160 → isolate bright text pixels (white/yellow game text)
      2. Compute per-row bright-pixel density
      3. Blank any row whose density < 42% of the median non-zero row density
         → these are typically sparse furigana rows
      4. 2× Lanczos upscale + 20 px black border

    Strengths : Very effective on Zelda-style white text on dark backgrounds.
    Weaknesses: Hard threshold at 160 misses darker or anti-aliased text;
                row blanking is pixel-column-agnostic (whole row goes blank).

    Returns white-on-black BGR image.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        h, w = crop.shape[:2]
        return np.zeros((h * 2 + 40, w * 2 + 40, 3), dtype=np.uint8)

    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)

    row_density = mask.sum(axis=1) / 255.0
    non_zero = row_density[row_density > 0]
    if len(non_zero) > 0:
        median_d = float(np.median(non_zero))
        furi_thresh = median_d * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furi_thresh:
                result[i, :] = 0

    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


def preprocess_clahe_adaptive(crop: np.ndarray) -> np.ndarray:
    """
    'CLAHE Adaptive' — Local contrast enhancement + adaptive threshold.

    Pipeline:
      1. Grayscale conversion
      2. CLAHE (clipLimit=2.0, tile=8×8) → equalises contrast in local patches,
         coping with dark game backgrounds that vary across the frame
      3. Adaptive Gaussian threshold (blockSize=15, C=8) → each pixel is
         thresholded relative to its local 15×15 neighbourhood; handles
         gradients that trip up a single global threshold
      4. Invert if majority of pixels are white (dark-text-on-light → white-on-black)
      5. 2× Lanczos upscale + 20 px black border

    Strengths : Works on both dark and light backgrounds without tuning;
                adaptive threshold survives spotlight/vignette backgrounds.
    Weaknesses: blockSize=15 can merge very small furigana glyphs with neighbours;
                CLAHE can amplify noise in very low-contrast regions.

    Returns white-on-black BGR image.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=15, C=8
    )

    # Normalise polarity: white text on black background
    if thresh.sum() > (thresh.size * 127):
        thresh = cv2.bitwise_not(thresh)

    result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


def preprocess_morph_sharpen(crop: np.ndarray) -> np.ndarray:
    """
    'Morph Sharpen' — Morphological background subtraction + stroke repair.

    Pipeline:
      1. Grayscale conversion
      2. Morphological CLOSE with a large (25×25) elliptical kernel estimates the
         slow-varying background illumination (game scene texture)
      3. background − foreground → isolates text as bright-on-dark even when the
         original contrast is low or the background is textured/gradient
      4. Normalise result to 0-255
      5. Otsu threshold → globally optimal binarisation on the clean signal
      6. Morphological CLOSE with 2×2 kernel → reconnects broken stroke pixels
         caused by anti-aliasing or compression artefacts
      7. 2× Lanczos upscale + 20 px black border

    Strengths : Robust against textured/gradient backgrounds (game scenes);
                stroke repair helps with small/compressed glyphs.
    Weaknesses: Large background kernel can bleed into large characters;
                subtraction flips polarity so dark text on light needs re-inversion.

    Returns white-on-black BGR image.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Estimate slow-varying background
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)

    # background - foreground lifts bright-on-dark text
    diff = cv2.subtract(background, gray).astype(np.float32)

    # Try the inverse too (for dark-on-light text)
    diff_inv = cv2.subtract(gray, background).astype(np.float32)

    # Pick whichever direction has stronger signal
    if diff_inv.max() > diff.max():
        diff = diff_inv

    d_min, d_max = diff.min(), diff.max()
    if d_max > d_min:
        normalized = ((diff - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(gray, dtype=np.uint8)

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Reconnect broken stroke pixels
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    repaired = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_repair)

    result = cv2.cvtColor(repaired, cv2.COLOR_GRAY2BGR)
    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


# ── Post-OCR Furigana Filtering ───────────────────────────────────────────────

def bimodal_furigana_filter(texts: list, heights: list, centres: list) -> list:
    """
    Bimodal gap split on bounding-box heights to separate furigana from main text.

    Uses the same algorithm as the original zelda_apple_ocr / zelda_paddle_ocr:
      • Find the largest gap in the sorted height distribution.
      • If that gap is >20% of the tallest box, treat everything below as furigana.
      • Isolation guard: keep small boxes whose vertical centre is within
        1.5×median_height of any main-text-line centre (small kana mid-line).

    Args:
        texts   : list of recognised text strings (one per box)
        heights : list of bounding-box pixel heights (same order)
        centres : list of bounding-box vertical centre positions (same order)

    Returns filtered list of text strings (furigana removed).
    """
    if not texts:
        return []

    sorted_h = sorted(heights)
    furi_thresh = sorted_h[0]

    if len(sorted_h) >= 2:
        gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
        max_gap, gap_idx = max(gaps)
        if max_gap > sorted_h[-1] * 0.20:
            furi_thresh = sorted_h[gap_idx + 1]

    median_h = float(np.median(sorted_h))
    large_centres = [c for h, c in zip(heights, centres) if h >= furi_thresh]

    filtered = []
    for t, h, cy in zip(texts, heights, centres):
        if h >= furi_thresh:
            filtered.append(t)
        elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
            filtered.append(t)
    return filtered


# ── Image Encoding ─────────────────────────────────────────────────────────────

def img_to_b64(img: np.ndarray) -> str:
    """Encode a BGR numpy image as a PNG base64 string for HTML data-URI embedding."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()
