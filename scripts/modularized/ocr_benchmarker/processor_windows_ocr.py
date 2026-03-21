"""
processor_windows_ocr.py
========================
OCR Processor: Windows.Media.Ocr (WinRT) — Japanese
Preprocessing: Row-density furigana suppression (Zelda Classic)

Windows.Media.Ocr is Microsoft's built-in on-device OCR API — the direct
Windows equivalent of Apple Vision. It runs fully locally (no API key, no
internet, no cost), uses hardware acceleration, and supports Japanese via
the OS language pack system.

Requirements:
  • Windows 10/11 only — raises NotImplementedError on macOS/Linux
  • Japanese language pack installed in Windows Settings
    Settings → Time & Language → Language & Region → Add Japanese
    Then install the "Basic typing" + "Optical character recognition" packs
  • Python package: pip install winrt-runtime winrt-Windows.Media.Ocr
                              winrt-Windows.Graphics.Imaging
                              winrt-Windows.Storage.Streams

On macOS/Linux this processor loads and benchmarks normally but returns a
clear unavailability message rather than crashing, so it can safely live in
the DEFAULT_PROCESSORS list on any platform.

Standalone usage (Windows only):
    python processor_windows_ocr.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import platform
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "Windows OCR (Japanese)"
DESCRIPTION = ("Windows.Media.Ocr WinRT · On-device · Japanese language pack · "
               "Row-density furigana suppression · Windows 10/11 only")

# ── Preprocessing + utils ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, img_to_b64

_IS_WINDOWS = platform.system() == "Windows"


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray) -> tuple:
    """
    Run Windows.Media.Ocr on a preprocessed BGR frame.
    Returns (text: str, elapsed_ms: int).

    WinRT OCR is async — we use asyncio.run() to drive it synchronously
    from the benchmark runner's synchronous context.

    Windows.Media.Ocr result structure:
        OcrResult.Lines  →  list of OcrLine
        OcrLine.Words    →  list of OcrWord
        OcrWord.Text     →  str

    We reconstruct lines by joining words, then apply bimodal furigana
    filtering using the bounding rect heights from OcrLine.

    Note on language selection:
        OcrEngine.TryCreateFromLanguage(Language("ja")) requires the
        Japanese OCR language pack to be installed in Windows Settings.
        If not installed, TryCreateFromLanguage returns None and we raise
        a clear RuntimeError explaining how to fix it.
    """
    if not _IS_WINDOWS:
        return "[Windows OCR unavailable — Windows 10/11 only]", 0

    import asyncio
    import winrt.windows.media.ocr as win_ocr
    import winrt.windows.globalization as globalization
    import winrt.windows.graphics.imaging as imaging
    import winrt.windows.storage.streams as streams
    import tempfile

    async def _ocr_async(img_path: str):
        # Load image via WinRT SoftwareBitmap
        with open(img_path, "rb") as f:
            data = f.read()

        mem_stream = streams.InMemoryRandomAccessStream()
        writer     = streams.DataWriter(mem_stream)
        writer.write_bytes(list(data))
        await writer.store_async()
        await writer.flush_async()
        mem_stream.seek(0)

        decoder    = await imaging.BitmapDecoder.create_async(mem_stream)
        bitmap     = await decoder.get_software_bitmap_async()

        lang       = globalization.Language("ja")
        engine     = win_ocr.OcrEngine.try_create_from_language(lang)
        if engine is None:
            raise RuntimeError(
                "Japanese OCR pack not installed. "
                "Go to Settings → Time & Language → Language & Region → "
                "Japanese → Language options → install 'Optical character recognition'."
            )

        result = await engine.recognize_async(bitmap)
        return result

    # Write frame to a temp PNG so WinRT can load it
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)

    try:
        t0     = time.perf_counter()
        result = asyncio.run(_ocr_async(tmp_path))
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        if not result or not result.lines:
            return "", elapsed_ms

        # Reconstruct lines and collect bounding box heights for furigana filter
        texts, heights, centres = [], [], []
        for line in result.lines:
            line_text  = " ".join(w.text for w in line.words)
            bbox       = line.bounding_rect          # Windows.Foundation.Rect
            y_min      = float(bbox.y)
            y_max      = float(bbox.y + bbox.height)
            texts.append(line_text)
            heights.append(y_max - y_min)
            centres.append((y_min + y_max) / 2.0)

        # Sort top-to-bottom
        order   = sorted(range(len(centres)), key=lambda i: centres[i])
        texts   = [texts[i]   for i in order]
        heights = [heights[i] for i in order]
        centres = [centres[i] for i in order]

        # Bimodal furigana filter
        from ocr_utils import bimodal_furigana_filter
        filtered = bimodal_furigana_filter(texts, heights, centres)

        return "\n".join(filtered), elapsed_ms

    finally:
        os.unlink(tmp_path)


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = preprocess_row_density(img)
    b64 = img_to_b64(preprocessed)

    if not _IS_WINDOWS:
        return {
            "text":             "[Windows OCR — runs on Windows 10/11 only]",
            "elapsed_ms":       0,
            "preprocessed_b64": b64,
        }

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except ImportError:
        text = (
            "[WinRT not installed — run: "
            "pip install winrt-runtime winrt-Windows.Media.Ocr "
            "winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams]"
        )
        elapsed_ms = 0
    except Exception as e:
        text       = f"[Windows OCR error: {e}]"
        elapsed_ms = 0

    return {"text": text, "elapsed_ms": elapsed_ms, "preprocessed_b64": b64}


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _IS_WINDOWS:
        print("Windows OCR is only available on Windows 10/11.")
        print("This processor will show as unavailable in benchmarks on other platforms.")
        sys.exit(0)

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
