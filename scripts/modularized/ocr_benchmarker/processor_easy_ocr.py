"""
processor_easy_ocr.py
=====================
OCR Processor: EasyOCR — Japanese language model
Preprocessing: Row-density furigana suppression (Zelda Classic)

EasyOCR uses a CRAFT text detector + CRNN recogniser, with separate
trained models per language. The Japanese model ('ja') was trained on
a mix of printed and digital Japanese text. Unlike RapidOCR's Chinese-
dominant training, EasyOCR's Japanese model is a first-class language
target with dedicated training data.

Key characteristics:
  • Separate detection (CRAFT) and recognition (CRNN) models
  • Japanese model handles hiragana, katakana, and kanji
  • Runs fully on CPU (MPS/GPU optional)
  • Bounding boxes available for post-OCR furigana filtering
  • First run downloads ~200MB model, cached after

Install:
    pip install easyocr

Standalone usage:
    python processor_easy_ocr.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import threading
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "EasyOCR (Japanese)"
DESCRIPTION = ("EasyOCR CRAFT+CRNN · Japanese language model · "
               "Row-density furigana suppression · Post-OCR bimodal filter")

# ── Preprocessing + utils ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, bimodal_furigana_filter, img_to_b64


# ── Lazy engine init ──────────────────────────────────────────────────────────

_engine    = None
_init_lock = threading.Lock()
_ocr_lock  = threading.Lock()


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _init_lock:
        if _engine is None:
            import easyocr
            # gpu=False for CPU-only; set gpu=True if CUDA/MPS available
            # ['ja'] loads the Japanese-specific model
            _engine = easyocr.Reader(['ja'], gpu=False, verbose=False)
    return _engine


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray) -> tuple:
    """
    Run EasyOCR on a preprocessed BGR frame.
    Returns (text: str, elapsed_ms: int).

    EasyOCR result format:
        [ ([tl, tr, br, bl], text, confidence), ... ]
        corner points: [[x,y], [x,y], [x,y], [x,y]]

    detail=True returns bounding boxes for furigana filtering.
    paragraph=False keeps individual line detections separate so we
    can apply the bimodal height filter before joining.
    """
    engine = _get_engine()

    t0 = time.perf_counter()
    with _ocr_lock:
        result = engine.readtext(
            frame,
            detail=1,          # return bounding boxes + confidence
            paragraph=False,   # keep lines separate for height filtering
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if not result:
        return "", elapsed_ms

    texts, heights, centres = [], [], []
    for (box, text, _conf) in result:
        pts   = np.array(box)          # [[x,y], [x,y], [x,y], [x,y]]
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        texts.append(text)
        heights.append(y_max - y_min)
        centres.append((y_min + y_max) / 2.0)

    # Sort top-to-bottom by vertical centre
    order   = sorted(range(len(centres)), key=lambda i: centres[i])
    texts   = [texts[i]   for i in order]
    heights = [heights[i] for i in order]
    centres = [centres[i] for i in order]

    # Bimodal furigana filter — same as all other processors
    filtered = bimodal_furigana_filter(texts, heights, centres)

    return "\n".join(filtered), elapsed_ms


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = preprocess_row_density(img)
    b64 = img_to_b64(preprocessed)

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except ImportError:
        text       = "[EasyOCR unavailable — pip install easyocr]"
        elapsed_ms = 0
    except Exception as e:
        text       = f"[EasyOCR error: {e}]"
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

    print(f"[{NAME}] Processing {len(paths)} image(s) in: {folder}")
    print("Note: First run downloads ~200MB model — subsequent runs use cache.\n")
    for p in paths:
        r = process_image(p)
        print(f"{'─'*60}")
        print(f"File  : {os.path.basename(p)}")
        print(f"Time  : {r['elapsed_ms']} ms")
        print(f"Text  : {r['text']}")
    print(f"{'─'*60}")
