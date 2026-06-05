# Calamity Ganon's Captions

Real-time Japanese dialogue translator and vocabulary trainer for Nintendo Switch games. Points a phone camera at your TV, reads the dialogue box, and gives you a live translation plus a full word-by-word breakdown to help you learn Japanese as you play.

Fully local — no cloud APIs or external services required. Works with any video source — phone camera, capture card, or webcam.

![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey) ![LLM](https://img.shields.io/badge/LLM-Ollama%20qwen3%3A8b-green) ![License](https://img.shields.io/badge/License-MIT-blue)

> This project was built through extensive iteration and experimentation — trying different OCR engines, vision models, LLM sizes, preprocessing approaches, and architectural patterns before arriving at the current design. The repo reflects the final state across three milestone versions. The full development story is documented in the [accompanying blog post](#). *(link coming soon)*

---

## Features

- **Translate mode** — live romaji + English translation as dialogue appears (~2.5s)
- **Learn mode** — word-by-word breakdown with readings, meanings, grammatical roles, and kanji analysis
- **Vocabulary tracking** — words colour-coded by familiarity (new / learning / familiar) based on exposure and quiz performance
- **Review quizzes** — triggered every N lessons, randomly sampled from recent vocabulary
- **All local** — runs entirely on your machine via Ollama, no data leaves your device
- **OS-native OCR** — Apple Vision on macOS, Windows.Media.Ocr on Windows; both are hardware-accelerated with no pip OCR packages required

---

## Requirements

**macOS**
- macOS (Apple Silicon — M1/M2/M3/M4)
- [Ollama](https://ollama.com) installed
- Python 3.9+
- A video source on the same network (phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam), capture card, or webcam)

**Windows**
- Windows 10/11 with Japanese language pack installed
  - Settings → Time & Language → Language & Region → Add Japanese
  - Windows 11 includes OCR automatically with the language pack
- [Ollama](https://ollama.com) installed
- Python 3.10+ (with "Add Python to PATH" checked during install)
- [DirectX 12 compatible GPU](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq) recommended (Windows OCR will GPU-accelerate automatically — RX 6800 included)
- A video source: capture card, webcam, or phone camera

---

## Quick Start

### macOS — Apple Vision OCR

```bash
chmod +x start_mac.sh     # first time only
./start_mac.sh
```

### Windows — Windows OCR

Double-click `start_windows.bat`, or from Command Prompt:

```bat
start_windows.bat
```

Both startup scripts handle everything automatically:

1. Check Python and install any missing dependencies
2. Check / install Ollama
3. Start the Ollama server if it isn't already running
4. Pull `qwen3:8b` if not already downloaded (~5GB on first run)
5. Run `calibrate.py` automatically if `bounds.json` is missing
6. Launch the translator

Once running, open **http://localhost:5002** in your browser.

---

## Why two startup scripts?

Each script is optimized for its OS's native OCR engine:

| | `start_mac.sh` | `start_windows.bat` |
|---|---|---|
| **OCR engine** | Apple Vision framework | Windows.Media.Ocr (WinRT) |
| **Script launched** | `zelda_apple_ocr.py` | `zelda_windows_ocr.py` |
| **GPU acceleration** | Apple Neural Engine / Metal | DirectML (DirectX 12) |
| **Extra setup** | None | Japanese language pack required |
| **OCR pip packages** | None (system framework) | None (system framework) |

Both target the same modularized script architecture under `scripts/modularized/` and share the same Ollama model (`qwen3:8b`), `bounds.json` crop config, and web UI.

> **`start_zelda_translator.sh`** is a legacy script from an earlier iteration of the project. It targets the old monolith at `scripts/zelda_translator_working_nlp.py` and pulls `qwen2.5:7b`. It is kept for reference but is no longer the recommended entry point — use `start_mac.sh` instead on macOS.

---

## Setup (Manual)

If you prefer not to use the startup scripts:

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/calamity-ganons-captions.git
cd calamity-ganons-captions
```

**2. Install dependencies**

macOS:
```bash
pip3 install opencv-python numpy requests flask \
             pyobjc-framework-Vision pyobjc-framework-Quartz \
             fugashi unidic-lite pykakasi jamdict jamdict-data
```

Windows:
```bash
pip install opencv-python numpy requests flask Pillow \
            fugashi unidic-lite pykakasi jamdict jamdict-data
```

**3. Start Ollama and pull the model**
```bash
ollama serve &          # or start Ollama.app on macOS
ollama pull qwen3:8b
```

**4. Calibrate the dialogue box crop (first time only)**

With your game running and a dialogue box visible on screen:
```bash
cd scripts/modularized
python3 calibrate.py    # macOS
python  calibrate.py    # Windows
```
Draw a rectangle over the dialogue box. Coordinates are saved to `bounds.json` and reused on every subsequent run.

**5. Run the translator**

macOS:
```bash
python3 scripts/modularized/zelda_apple_ocr.py
```

Windows:
```bash
python scripts\modularized\zelda_windows_ocr.py
```

**6. Open the UI**

Navigate to `http://localhost:5002` in your browser.

---

## Usage

- **Translate tab** — always live. Shows romaji and English translation as dialogue appears.
- **Learn tab** — generates a full lesson for each dialogue line. Hit **Got it** to acknowledge, save vocab, and unlock the next lesson. Toggle Learn mode on or off from the UI.
- Quizzes trigger automatically every N lessons. `QUIZ_EVERY` in the script config controls the frequency.

---

## File Structure

```
start_mac.sh                            # macOS startup script (Apple Vision OCR)
start_windows.bat                       # Windows startup script (Windows OCR)
start_zelda_translator.sh               # Legacy startup script (kept for reference)

scripts/
  modularized/
    zelda_apple_ocr.py                  # Main script — macOS (Apple Vision)
    zelda_windows_ocr.py                # Main script — Windows (Windows.Media.Ocr)
    zelda_core.py                       # Shared core logic
    calibrate.py                        # One-time setup: draw crop bounds
    bounds.json                         # Generated by calibrate.py (macOS)
    macos_bounds.json                   # macOS bounds reference
    windows_bounds.json                 # Windows bounds reference
    live_viewer.py                      # Debug: live OCR region viewer
    ocr_benchmarker/                    # OCR engine benchmarking tools
    av/                                 # Apple Vision training data and metrics
    paddle_*/                           # PaddleOCR pipeline variants (benchmarking)

  monolith/                             # Legacy single-file scripts (kept for reference)
    zelda_translator_working_nlp.py     # Original NLP hybrid monolith
    zelda_translator_*.py               # Earlier iteration variants

iterative-scripts/                      # Early R&D scripts (kept for reference)
  working-apps/                         # Milestone working versions
```

---

## Stack

| Component | Technology |
|---|---|
| Camera / video feed | Any MJPEG source — phone + IP Webcam app, capture card, or webcam |
| OCR (macOS) | Apple Vision framework (hardware-accelerated, Neural Engine / Metal) |
| OCR (Windows) | Windows.Media.Ocr via WinRT (hardware-accelerated, DirectML) |
| Word segmentation | fugashi (MeCab) |
| Romaji | pykakasi |
| Dictionary | jamdict (JMdict + Kanjidic) |
| Translation | qwen3:8b via Ollama |
| Web UI | Flask |

---

## License

MIT
