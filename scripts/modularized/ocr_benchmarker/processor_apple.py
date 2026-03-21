"""
processor_apple.py
==================
OCR Processor: Apple Vision Framework (macOS only)
Preprocessing: Row-density furigana suppression (Zelda Classic)

Benchmark baseline — mirrors zelda_apple_ocr.py exactly.

Standalone usage:
    python processor_apple.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "Apple Vision"
DESCRIPTION = "Apple Vision OCR (macOS) · Row-density furigana suppression · Benchmark"

# ── Preprocessing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, img_to_b64


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray):
    """
    Run Apple Vision on a preprocessed BGR frame.
    Returns (text: str, elapsed_ms: int).
    Raises ImportError on non-macOS systems.
    """
    import tempfile
    import Vision
    import Quartz

    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)

    try:
        img_url  = Quartz.CFURLCreateFromFileSystemRepresentation(
                       None, tmp_path.encode(), len(tmp_path), False)
        src      = Quartz.CGImageSourceCreateWithURL(img_url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

        raw_observations = []

        def _handler(request, error):
            if error:
                return
            for obs in request.results():
                cand = obs.topCandidates_(1)
                if cand:
                    raw_observations.append((cand[0].string(), obs.boundingBox()))

        req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(_handler)
        req.setRecognitionLanguages_(["ja"])
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc() \
            .initWithCGImage_options_(cg_image, {}) \
            .performRequests_error_([req], None)

        img_h = frame.shape[0]
        candidates = []
        for text, bbox in raw_observations:
            px_h  = bbox.size.height * img_h
            top_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
            cy    = top_y + px_h / 2.0
            candidates.append((text, px_h, cy))

        if not candidates:
            return "", round((time.perf_counter() - t0) * 1000)

        # Bimodal gap split (furigana filter on Vision bounding boxes)
        sorted_h    = sorted(h for _, h, _ in candidates)
        furi_thresh = sorted_h[0]
        if len(sorted_h) >= 2:
            gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > sorted_h[-1] * 0.20:
                furi_thresh = sorted_h[gap_idx + 1]

        median_h      = float(np.median(sorted_h))
        large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

        texts = []
        for text, px_h, cy in candidates:
            if px_h >= furi_thresh:
                texts.append(text)
            elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                texts.append(text)

        return " ".join(texts).strip(), round((time.perf_counter() - t0) * 1000)

    finally:
        os.unlink(tmp_path)


# ── Public interface (required by ocr_benchmark.py) ──────────────────────────

def process_image(img_path: str) -> dict:
    """
    Process a single image file.

    Returns:
        dict with keys:
            text           (str)  — recognised Japanese text
            elapsed_ms     (int)  — wall-clock time for OCR only
            preprocessed_b64 (str) — base64 PNG of preprocessed image (for report)
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = preprocess_row_density(img)
    b64 = img_to_b64(preprocessed)

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except ImportError:
        text       = "[Apple Vision unavailable — macOS only]"
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
