# Capy - Development Guide

## Cursor Cloud specific instructions

### Overview

Capy is a local AI assistant CLI application (Python 3.10+). It provides LLM inference via `llama-cpp-python` and optional voice input via `faster-whisper` + `pyaudio`. See `README.md` for feature list and quick-start.

### Conda environment

Always activate the `duc_learning` conda environment before running any Python commands:

```bash
conda activate duc_learning
```

Conda is installed at `~/miniconda3`. The shell profile (`.bashrc`) is configured to auto-activate `duc_learning` on login.

### Running the application

- **Text-mode chat** (no microphone needed): use `capy/chat_engine.py` — the simpler engine without ASR.
- **Full mode with voice** (requires a real audio device): use `capy/core/chat_engine.py` via `python test_capy.py`.
- The LLM model file lives at `./models/Llama-3.2-3B-Instruct-Q5_K_S.gguf` (~2.2 GB). It is downloaded from `bartowski/Llama-3.2-3B-Instruct-GGUF` on Hugging Face.

### Cloud VM caveats

- **No audio hardware**: The cloud VM has no microphone. `capy.core.chat_engine.CapyChatEngine` will fail to instantiate because `CapyASR.__init__` opens a PyAudio stream. Use the simpler `capy.chat_engine.CapyChatEngine` for text-only testing in the cloud.
- **ALSA warnings**: PyAudio prints many ALSA "cannot find card" warnings — these are harmless in a headless environment.
- The Whisper `tiny.en` model is pre-cached via `faster_whisper`; the ASR code uses `local_files_only=True`.

### Linting and testing

- No formal test framework is configured. `test_capy.py` is a manual integration test (interactive CLI).
- `ruff check .` passes cleanly and can be used for linting.
- All Python files compile cleanly (`python -m py_compile <file>`).

### System dependencies

- `portaudio19-dev` — required to build/install `pyaudio`.
- `cmake`, `gcc` — required to build `llama-cpp-python` from source.
