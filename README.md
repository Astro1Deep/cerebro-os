# Cerebro OS

[Español](README_ES.md) · [User guide](docs/GUIDE.md) · [Release notes](docs/RELEASE_NOTES.md)

Cerebro OS is a local-first workstation for language models, music composition and library search. This repository contains the portable **1.5.0-beta.3** source. Composer works offline for chord analysis, progressions and MIDI. Jarvis connects to a local model for text chat and can read connected BeatOS indexes.

| Component | What it does |
|---|---|
| Composer | Chord analysis, scale and Push grids, progressions, browser pitch detection, MIDI import/export |
| Jarvis | Text chat, local diagnostics and retrieval from BeatOS catalog indexes |
| Model server | MLX-LM on Apple Silicon or an Ollama server with a local model |
| CLI | Service status, offline diagnostics, profiles, local knowledge and Apple Silicon model evaluation |
| Telegram | Optional private-chat bridge with explicit access policy |

Start with Python 3.12 and install into a new directory. The guide covers macOS, Linux and Windows commands, the model endpoints, connection options and beta validation status.

```sh
python3 install.py --target ~/cerebro-beta
```
