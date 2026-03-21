"""
processor_manga_ocr_colour.py
=============================
OCR Processor: MangaOCR — Colour crop variant (no binarisation)
Preprocessing: Raw colour crop passed directly, no thresholding

MangaOCR was trained on colour manga panels. Passing a binarised white-on-black
image (as processor_manga_ocr.py does) may hurt accuracy because the model
expects colour/grayscale input. This variant passes the raw colour crop with
only a 2× upscale — no threshold, no row-density suppression.

The tradeoff: furigana is NOT pre-suppressed here, so the model may read
furigana inline with the main text. MangaOCR's training data includes
furigana so it often ignores it naturally, but this needs benchmarking.

Run alongside processor_manga_ocr.py to compare binarised vs colour input.

Install:
    pip install manga-ocr

Standalone usage:
    python processor_manga_ocr_colour.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import threading
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "MangaOCR (colour)"
DESCRIPTION = ("MangaOCR vision-transformer · Raw colour crop · 2× upscale only · "
               "No binarisation · Furigana handled by model training")

# ── Preprocessing + utils ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import img_to_b64


# ── Lazy engine init ──────────────────────────────────────────────────────────

_engine    = None
_init_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _init_lock:
        if _engine is None:
            from manga_ocr import MangaOcr
            _engine = MangaOcr()
    return _engine


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_colour(crop: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for MangaOCR colour mode.
    Only upscales 2× with Lanczos — no thresholding, no binarisation.
    Preserves the colour information MangaOCR was trained on.
    """
    h, w = crop.shape[:2]
    result = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    return result


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray) -> tuple:
    """
    Run MangaOCR on a colour BGR frame.
    Converts BGR → RGB PIL for MangaOCR.
    Returns (text: str, elapsed_ms: int).
    """
    from PIL import Image as PILImage

    engine = _get_engine()

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    t0 = time.perf_counter()
    text = engine(pil_img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return text.strip(), elapsed_ms


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = _preprocess_colour(img)
    b64 = img_to_b64(preprocessed)

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except ImportError:
        text       = "[MangaOCR unavailable — pip install manga-ocr]"
        elapsed_ms = 0
    except Exception as e:
        text       = f"[MangaOCR error: {e}]"
        elapsed_ms = 0

    return {"text": text, "elapsed_ms": elapsed_ms, "preprocessed_b64": b64}


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    paths  = sorted(
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )
    if not paths:
        print(f"No images found in: {folder}")
        sys.exit(1)

    print(f"[{NAME}] Processing {len(paths)} image(s) in: {folder}\n")
    for p in paths:
        r = process_image(p)
        print(f"{'─'*60}")
        print(f"File  : {os.path.basename(p)}")
        print(f"Time  : {r['elapsed_ms']} ms")
        print(f"Text  : {r['text']}")
    print(f"{'─'*60}")
