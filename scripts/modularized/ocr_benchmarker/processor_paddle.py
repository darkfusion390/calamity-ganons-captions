"""
processor_paddle.py
===================
OCR Processor: PaddleOCR v5 Mobile (PP-OCRv5)
Preprocessing: Row-density furigana suppression (Zelda Classic)

Benchmark baseline — mirrors zelda_paddle_ocr.py exactly.

Install:
    pip install paddlepaddle paddleocr

Standalone usage:
    python processor_paddle.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import threading
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "PaddleOCR v5 (3x upscale)"
DESCRIPTION = "PaddleOCR PP-OCRv5 mobile · Row-density furigana suppression · 3× Lanczos upscale"

# ── Preprocessing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, bimodal_furigana_filter, img_to_b64


def _preprocess_paddle_3x(crop: np.ndarray) -> np.ndarray:
    """
    Paddle-specific variant of row-density furigana suppression.
    Identical pipeline to preprocess_row_density() in ocr_utils, but uses a
    3× Lanczos upscale instead of 2× — Paddle's mobile det model benefits from
    higher resolution input and was dropping characters at 2×.
    """
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        h, w = crop.shape[:2]
        return np.zeros((h * 3 + 40, w * 3 + 40, 3), dtype=np.uint8)

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
    result = cv2.resize(result, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


# ── Lazy engine init (avoid 2-second cold start unless actually used) ──────────

_paddle_ocr  = None
_paddle_lock = threading.Lock()
_init_lock   = threading.Lock()


def _get_engine():
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    with _init_lock:
        if _paddle_ocr is None:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
            )
    return _paddle_ocr


# ── OCR engine ────────────────────────────────────────────────────────────────

def _postprocess_paddle(pairs: list) -> list:
    """
    Apply targeted fixes and drop noise lines.
    Exact-string substitution table left empty — add verified entries as found.
    """
    _EXACT_FIXES = {}

    def _fix_hira_before_kata_N(text: str) -> str:
        result = list(text)
        for i in range(len(result) - 1):
            if 'ぁ' <= result[i] <= 'ん' and result[i + 1] == 'ン':
                result[i] = chr(ord(result[i]) + 0x60)
        return ''.join(result)

    out = []
    for t, s in pairs:
        for wrong, correct in _EXACT_FIXES.items():
            t = t.replace(wrong, correct)
        t = _fix_hira_before_kata_N(t)
        if len(t.strip()) > 3:
            out.append((t, s))
    return out


def _run_ocr(frame: np.ndarray):
    """
    Run PaddleOCR on a preprocessed BGR frame.
    Returns (text: str, elapsed_ms: int).
    """
    engine = _get_engine()

    t0 = time.perf_counter()
    with _paddle_lock:
        result = engine.predict(frame)

    all_texts, all_scores, all_heights, all_centres = [], [], [], []
    for res in (result or []):
        polys  = res.get("rec_polys") or res.get("rec_boxes") or []
        t_list = res.get("rec_texts") or []
        s_list = res.get("rec_scores") or []
        for poly, t, s in zip(polys, t_list, s_list):
            pts   = np.array(poly)
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            all_texts.append(t)
            all_scores.append(s)
            all_heights.append(y_max - y_min)
            all_centres.append((y_min + y_max) / 2.0)

    # Sort top-to-bottom
    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres = map(list, zip(*combined))

    # Bimodal furigana filter on bounding-box heights
    filtered_texts = bimodal_furigana_filter(all_texts, all_heights, all_centres)
    filtered_scores = [s for t, s in zip(all_texts, all_scores) if t in filtered_texts]

    # Post-processing fixes + noise filter
    pairs = _postprocess_paddle(list(zip(filtered_texts, filtered_scores)))
    texts = [t for t, _ in pairs]

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return "\n".join(texts), elapsed_ms


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = _preprocess_paddle_3x(img)
    b64 = img_to_b64(preprocessed)

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except Exception as e:
        text       = f"[PaddleOCR error: {e}]"
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
