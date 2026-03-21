# OCR Benchmark Suite

Benchmark harness for comparing Japanese OCR engines and preprocessing pipelines
against Zelda: Breath of the Wild dialogue captures (and other game/UI text images).

---

## File Structure

```
ocr_suite/
├── ocr_benchmark.py        ← CORE: run all processors, generate HTML report
├── ocr_utils.py            ← shared preprocessing + filtering utilities
│
├── processor_apple.py      ← Apple Vision OCR       (macOS only, benchmark baseline)
├── processor_paddle.py     ← PaddleOCR v5 mobile    (benchmark baseline)
│
├── processor_rapid_a.py    ← RapidOCR  · Approach A: Row-Density (same as baselines)
├── processor_rapid_b.py    ← RapidOCR  · Approach B: CLAHE + Adaptive Threshold
└── processor_rapid_c.py    ← RapidOCR  · Approach C: Morph Background Subtraction
```

---

## Install

```bash
# Required for all RapidOCR processors
pip install rapidocr-onnxruntime opencv-python numpy

# PaddleOCR baseline
pip install paddlepaddle paddleocr

# Apple Vision: built-in on macOS, no pip install needed
# Requires: pyobjc-framework-Vision pyobjc-framework-Quartz
```

---

## Usage

### Run full benchmark (all processors)
```bash
python ocr_benchmark.py /path/to/images/folder
```

### Run specific processors only
```bash
python ocr_benchmark.py /path/to/images --processors processor_rapid_a processor_rapid_b processor_rapid_c
```

### Custom output path
```bash
python ocr_benchmark.py /path/to/images --output /path/to/my_report.html
```

### Run a single processor standalone (for quick iteration)
```bash
python processor_rapid_b.py /path/to/images/folder
```

---

## Preprocessing Approaches

| Processor       | Preprocessing                        | Furigana removal   | Best for                               |
|-----------------|--------------------------------------|--------------------|----------------------------------------|
| Apple Vision    | Row-density threshold (T=160)        | Pre-OCR row blank  | Zelda-style white text on dark bg      |
| PaddleOCR v5    | Row-density threshold (T=160)        | Pre-OCR row blank  | Zelda-style white text on dark bg      |
| Rapid-A         | Row-density threshold (T=160)        | Pre + post bimodal | Engine comparison (same input as above)|
| Rapid-B         | CLAHE → Adaptive Gaussian (block=15) | Post-OCR bimodal   | Mixed bg brightness, varied scenes     |
| Rapid-C         | Morph background subtract → Otsu     | Post-OCR bimodal   | Textured/gradient game backgrounds     |

---

## Adding a New Processor

1. Create `processor_myengine.py` with this interface:

```python
NAME        = "My Engine"          # display name
DESCRIPTION = "one-line summary"   # shown in report

def process_image(img_path: str) -> dict:
    # Must return:
    return {
        "text":             str,   # recognised text
        "elapsed_ms":       int,   # wall-clock OCR time
        "preprocessed_b64": str,   # base64 PNG of preprocessed image
    }
```

2. Add `"processor_myengine"` to `DEFAULT_PROCESSORS` in `ocr_benchmark.py`,
   or pass it via `--processors processor_myengine`.

---

## HTML Report

The report is self-contained (all images embedded as base64 data URIs) and includes:

- **Timing summary table** — ms per processor per image
- **Per-image sections** showing:
  - Original image
  - Each processor's preprocessed image (so you can visually inspect what the OCR saw)
  - Extracted text output
  - Elapsed time

No ground-truth labels are required — comparison is visual/manual.
If you add a `ground_truth/` folder with `.txt` files matching image names,
a future version could compute CER/WER automatically.
