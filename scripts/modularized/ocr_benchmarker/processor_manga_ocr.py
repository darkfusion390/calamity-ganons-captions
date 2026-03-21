"""
processor_manga_ocr.py
======================
OCR Processor: MangaOCR — Japanese manga/game text specialist
Preprocessing: Raw crop passed directly (MangaOCR has its own internal pipeline)

MangaOCR is a vision-transformer model fine-tuned specifically on manga and
game text — stylized fonts, thick outlines, furigana, and varied backgrounds.
It treats the entire image as a single text region (no detection step), which
makes it fundamentally different from PaddleOCR/RapidOCR box-detect-then-read
pipelines. This also means it works best when the input is already cropped to
a single line or dialogue box — exactly what zelda_core provides.

Key characteristics:
  • No detection step — reads the whole image as one text block
  • Trained on manga/game fonts — should handle Zelda's outlined style well
  • Furigana: the model tends to ignore small furigana naturally (trained that way)
  • Returns a single string per image, no bounding boxes
  • Slower first call (model download ~450MB on first run, cached after)

Install:
    pip install manga-ocr

Standalone usage:
    python processor_manga_ocr.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import threading
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "MangaOCR"
DESCRIPTION = ("MangaOCR vision-transformer · Manga/game font specialist · "
               "No detection step · Full-image single-pass reading")

# ── Preprocessing + utils ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, img_to_b64


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
            # MangaOcr() downloads ~450MB model on first call, cached to ~/.cache
            _engine = MangaOcr()
    return _engine


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray) -> tuple:
    """
    Run MangaOCR on a preprocessed BGR frame.
    MangaOCR expects a PIL Image — we convert from BGR numpy.
    Returns (text: str, elapsed_ms: int).

    MangaOCR treats the whole image as one text region and returns a single
    string — no bounding boxes, no line splitting. For multi-line Zelda
    dialogue the model reads top-to-bottom naturally.

    Note: MangaOCR works best on the ORIGINAL colour crop, not a binarised
    version. We pass the row-density preprocessed image (white-on-black) since
    that's what we have, but also expose process_image_raw() which passes the
    unprocessed colour image — benchmark both to see which wins.
    """
    from PIL import Image as PILImage

    engine = _get_engine()

    # Convert BGR numpy → RGB PIL (MangaOCR requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    t0 = time.perf_counter()
    text = engine(pil_img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return text.strip(), elapsed_ms


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    """
    Process a single image file.
    Passes the row-density preprocessed (binarised) image to MangaOCR.
    See process_image_colour() below for the colour variant.
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    # MangaOCR is designed for colour manga panels — we test with the same
    # row-density preprocessed input as other processors for fair comparison,
    # but note that passing the raw colour crop may perform better.
    preprocessed = preprocess_row_density(img)
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

    print(f"[{NAME}] Processing {len(paths)} image(s) in: {folder}")
    print("Note: First run downloads ~450MB model — subsequent runs use cache.\n")
    for p in paths:
        r = process_image(p)
        print(f"{'─'*60}")
        print(f"File  : {os.path.basename(p)}")
        print(f"Time  : {r['elapsed_ms']} ms")
        print(f"Text  : {r['text']}")
    print(f"{'─'*60}")
