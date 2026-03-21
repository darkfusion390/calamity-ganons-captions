# Calamity Ganon's Captions

Real-time Japanese dialogue translator and vocabulary trainer for Nintendo Switch games. Points a phone camera or capture card at your TV, reads the dialogue box, and gives you a live translation plus a full word-by-word breakdown to help you learn Japanese as you play.

Fully local — no cloud APIs and no dependence on external services. Works with any video source — phone camera, capture card, or webcam. Runs on both macOS and Windows.

![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey) ![LLM](https://img.shields.io/badge/LLM-Ollama%20qwen3%3A8b-green) ![License](https://img.shields.io/badge/License-MIT-blue)

> This project was built through extensive iteration and experimentation — trying different OCR engines, vision models, LLM sizes, preprocessing approaches, and architectural patterns before arriving at the current design. The full development story is documented in the [accompanying blog post](#). *(link coming soon)*

---

## Features

- **Translate mode** — live romaji + English translation as dialogue appears (~1.5–2.5s on macOS, ~2.5–3.5s on Windows)
- **Learn mode** — word-by-word breakdown with readings, meanings, grammatical roles, and kanji analysis
- **Vocabulary tracking** — words colour-coded by familiarity (new / learning / familiar) based on exposure and quiz performance
- **Review quizzes** — triggered every N lessons, randomly sampled from recent vocabulary
- **Multi-region OCR** — simultaneously reads multiple screen regions (dialogue box, item title, item description, speaker name) and selects the active group by Japanese character count
- **Cross-platform** — macOS (Apple Vision OCR on Neural Engine) and Windows (Windows.Media.Ocr via PowerShell, DirectML-accelerated)
- **All local** — runs entirely on your machine via Ollama, no data leaves your device

---

## Requirements

**macOS:**
- macOS Apple Silicon (M1/M2/M3/M4)
- [Ollama](https://ollama.com) installed
- Python 3.9+
- Video source: phone camera via IP Webcam app, USB webcam, or capture card

**Windows:**
- Windows 10/11
- [Ollama](https://ollama.com) installed
- Python 3.10+
- Japanese language pack: Settings → Time & Language → Language & Region → Add Japanese
- Video source: USB webcam or capture card (DirectShow)

---

## Quick Start

```bash
# macOS
chmod +x start_mac.sh
./start_mac.sh

# Windows
start_windows.bat
```

Both scripts handle the full setup automatically: dependency checks and installation, Ollama setup and model download, calibration if bounds.json is missing, and launching the correct OCR backend for your platform.

Open `http://localhost:5002` in your browser once running.

---

## Final Scripts

The project has converged on two platform-specific entry points that share the same `zelda_core.py` engine:

### `zelda_apple_ocr.py` — macOS ⭐ Recommended (macOS)

Uses Apple Vision framework — runs on the M1/M2/M3/M4 Neural Engine with zero RAM footprint and ~175ms wall-clock OCR across 4 regions concurrently. The unambiguous winner in every accuracy and latency benchmark.

```bash
pip install opencv-python numpy requests flask pyobjc-framework-Vision pyobjc-framework-Quartz
pip install fugashi unidic-lite pykakasi jamdict jamdict-data
ollama pull qwen3:8b
```

### `zelda_windows_ocr.py` — Windows ⭐ Recommended (Windows)

Uses Windows.Media.Ocr via PowerShell — no pip OCR packages required. Runs via Windows ML / DirectML, which leverages GPU acceleration on any DirectX 12 compatible card automatically. Benchmarked at 88.5% character accuracy across 13 test images, on par with EasyOCR and RapidOCR with the Japanese model.

```bash
pip install opencv-python numpy requests flask fugashi unidic-lite pykakasi jamdict jamdict-data Pillow
ollama pull qwen3:8b
```

**Windows also requires:** Japanese language pack installed (Settings → Time & Language → Language & Region → Add Japanese).

---

## Earlier Versions (Historical Reference)

These live in `iterative-scripts/working-apps/` and represent earlier stages of development.

### `zelda_translator_working_apple_OCR.py` — Translation only
The simplest version. OCR extracts Japanese text, LLM translates it to English. No learn mode, no vocab tracking, no quizzes.

### `zelda_translator_working_apple_OCR_learning.py` — Translation + Learn mode (LLM only)
Adds a full Learn tab. The LLM handles everything in Learn mode in a single prompt. Learn mode takes 17,000–27,000ms per lesson — this was the predecessor to the NLP version.

### `zelda_translator_working_nlp.py` — Translation + Learn mode (NLP hybrid)
The monolith predecessor to the current modular design. Learn mode rebuilt with fugashi/pykakasi/jamdict, dropping lesson generation from 17–27s to ~2.7s. All functionality now lives in `zelda_core.py`.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/darkfusion390/calamity-ganons-captions.git
cd calamity-ganons-captions
```

**2. Run the startup script (recommended)**
```bash
# macOS
chmod +x start_mac.sh && ./start_mac.sh

# Windows
start_windows.bat
```

The startup scripts check all dependencies, install anything missing, start Ollama, pull the model, auto-run calibrate.py if bounds.json is missing, and launch the correct OCR backend.

**3. Calibrate crop regions (first time only)**

If running manually: with your game running and a dialogue box visible:
```bash
cd scripts/modularized
python3 calibrate.py
```

Draw rectangles over each text region (dialogue box, item title, item description, speaker name). Group related regions so they're evaluated together. Saved to `bounds.json`.

**4. Open the UI**

Navigate to `http://localhost:5002` in your browser.

---

## Usage

- **Translate tab** — always live. Shows romaji and English translation as dialogue appears.
- **Learn tab** — generates a full lesson for each dialogue line. Hit **Got it** to acknowledge, save vocab, and unlock the next lesson.
- Quizzes trigger automatically every N lessons (`QUIZ_EVERY` in config).
- Preview streams at `/preview/<group_name>` let you inspect what each region is seeing.

---

## File Structure

```
scripts/
├── modularized/
│   ├── zelda_core.py              # Shared engine: OCR loop, gates, NLP, Flask UI
│   ├── zelda_apple_ocr.py         # macOS: Apple Vision OCR backend ⭐
│   ├── zelda_windows_ocr.py       # Windows: Windows.Media.Ocr backend ⭐
│   ├── zelda_paddle_ocr.py        # PaddleOCR backend (experimental)
│   ├── calibrate.py               # Multi-region calibration tool
│   └── ocr_benchmarker/           # Standalone OCR accuracy benchmark suite
│       ├── ocr_benchmark.py       # Core runner — generates HTML accuracy report
│       ├── processor_apple.py
│       ├── processor_paddle.py
│       ├── processor_rapid_a/b/c.py
│       ├── processor_manga_ocr.py
│       ├── processor_manga_ocr_colour.py
│       ├── processor_easy_ocr.py
│       └── processor_windows_ocr.py
├── monolith/                      # Earlier single-file versions (historical)
└── ...
iterative-scripts/                 # Experimental scripts from development
start_mac.sh                       # macOS startup script
start_windows.bat                  # Windows startup script
```

---

## Stack

| Component | macOS | Windows |
|---|---|---|
| Video input | IP Webcam / capture card / webcam | Capture card / webcam (DirectShow) |
| OCR | Apple Vision (Neural Engine) | Windows.Media.Ocr (DirectML) |
| OCR concurrency | ThreadPoolExecutor, one thread per region | Parallel PowerShell subprocesses |
| Word segmentation | fugashi (MeCab) | fugashi (MeCab) |
| Romaji | pykakasi | pykakasi |
| Dictionary | jamdict (JMdict + Kanjidic) | jamdict (JMdict + Kanjidic) |
| Translation | qwen3:8b via Ollama | qwen3:8b via Ollama |
| Web UI | Flask | Flask |

---

## OCR Benchmark Results

A structured benchmark suite (`ocr_benchmarker/`) was built to compare engines objectively on real BotW gameplay captures. Results across 13 images (character error rate, lower is better):

| Engine | Avg Accuracy | Avg OCR Latency | Notes |
|---|---|---|---|
| Apple Vision | ~99% (benchmark baseline) | ~175ms (4 regions, parallel) | Neural Engine, zero RAM cost |
| RapidOCR-A (JA PPOCRv4) | 88.9% | ~388ms | Japanese model required |
| EasyOCR (Japanese) | 88.6% | ~2,800ms | Slow but solid |
| Windows OCR | 88.5% | ~255ms | DirectML GPU-accelerated |
| PaddleOCR v5 mobile | 84.1% | ~955ms | Good on single-column text |
| MangaOCR (colour) | 72.6% | ~650ms | Struggles on dark game backgrounds |
| MangaOCR (binarised) | 69.5% | ~600ms | Trained on white-background manga |

Apple Vision is the clear production winner on macOS. On Windows, Windows OCR offers the best speed/accuracy balance of the cross-platform engines.

---

## License

MIT
