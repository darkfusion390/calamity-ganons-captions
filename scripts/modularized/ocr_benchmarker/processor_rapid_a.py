"""
processor_rapid_a.py
====================
OCR Processor: RapidOCR — Preprocessing: Row-Density Furigana Suppression
Approach A: "Zelda Classic" — same preprocessing as Apple/Paddle benchmarks.

Goal: Isolate the effect of the OCR *engine* while holding preprocessing constant.
      Compare this output directly against processor_apple and processor_paddle
      to see how RapidOCR performs on the identical input.

Install:
    pip install rapidocr

Standalone usage:
    python processor_rapid_a.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import threading
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "RapidOCR-A (JA PPOCRv4 · Row Density)"
DESCRIPTION = ("RapidOCR · Japanese PPOCRv4 server rec · Row-density furigana suppression · "
               "Engine comparison vs Apple/Paddle")

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
            from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
            # LangRec.JAPAN only exists for PPOCRv4 — v5 has no Japanese model yet.
            # ModelType.SERVER is more accurate than MOBILE at the cost of ~30MB model.
            _engine = RapidOCR(params={
                "Rec.lang_type":   LangRec.JAPAN,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Rec.model_type":  ModelType.SERVER,
            })
    return _engine


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray):
    """
    Run RapidOCR on a preprocessed BGR frame.
    Applies post-OCR bimodal furigana filter on bounding-box heights.
    Returns (text: str, elapsed_ms: int).

    RapidOCR result format:
        [[box_points, text, confidence], ...]
        box_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (or ndarray)
    """
    engine = _get_engine()

    t0 = time.perf_counter()
    with _ocr_lock:
        res = engine(frame)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # rapidocr v3 returns a RapidOCROutput object — res.boxes is a numpy array,
    # so must use len() rather than bare bool test to avoid "truth value of array" error
    boxes = getattr(res, "boxes", None)
    txts  = getattr(res, "txts",  None)
    if boxes is None or len(boxes) == 0:
        return "", elapsed_ms

    texts, heights, centres = [], [], []
    for box, text in zip(boxes, txts):
        pts   = np.array(box)
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        texts.append(text)
        heights.append(y_max - y_min)
        centres.append((y_min + y_max) / 2.0)

    # Sort top-to-bottom
    order   = sorted(range(len(centres)), key=lambda i: centres[i])
    texts   = [texts[i]   for i in order]
    heights = [heights[i] for i in order]
    centres = [centres[i] for i in order]

    # Bimodal furigana filter (shared utility)
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
    except Exception as e:
        text       = f"[RapidOCR error: {e}]"
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
