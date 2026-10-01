# VoiceStudio

> **Local-first voice cloning, TTS, dubbing, and audiobook creation**

A modern, Electron-based desktop application for speech synthesis with a multi-engine FastAPI backend. Built as a spiritual successor to the original Speech Synthesis project, mirroring the architecture of [VoiceStudio](https://github.com/debpalash/VoiceStudio).

## Features

- **Voice Cloning** - Clone any voice from a 3-15 second reference sample
- **Voice Design** - Create voices from natural language descriptions
- **Video Dubbing** - Dub videos with synchronized speech (single & batch)
- **Stories & Audiobooks** - Long-form content creation with multi-speaker support
- **Transcription** - ASR with speaker diarization
- **Local-First** - Runs entirely on your hardware (CUDA, MPS, CPU)
- **Multi-Engine** - Pluggable TTS/ASR/Vocoder engines via subprocess isolation
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI TTS/STT endpoints
- **MCP Server** - Model Context Protocol integration for AI agents

## Architecture

```
speech-synthesis-studio/
├── electron/                 # Electron desktop app (React + TypeScript + Tailwind v4)
│   ├── src/
│   │   ├── main/            # Main process (backend supervision, app protocol)
│   │   ├── preload/         # Preload script (secure IPC bridge)
│   │   └── renderer/        # React renderer (workspaces, components, hooks)
│   └── package.json
├── backend/                  # FastAPI backend (Python)
│   ├── api/                 # REST API routes
│   ├── core/                # Config, auth, subprocess management
│   ├── engines/             # Engine sidecars (subprocess-isolated)
│   ├── services/            # Model management, generation
│   └── pyproject.toml
├── omnivoice/                # Core models package
│   ├── models/              # Model architectures
│   ├── training/            # Training pipelines
│   └── pyproject.toml
├── scripts/                  # Development scripts
└── package.json              # Root monorepo config (bun + turbo)
```

## Quick Start

### Prerequisites

- **Node.js** 22+ (or [Bun](https://bun.sh) 1.4+)
- **Python** 3.11+
- **uv** (Python package manager) - `pip install uv`
- **Git**

### Development

```bash
# Clone and install
git clone <your-repo>
cd speech-synthesis-studio

# Install all dependencies (Node + Python)
bun install
bun run setup:api

# Start development (backend + frontend + Electron)
bun run dev

# Or start web preview only (no Electron)
bun run dev:web
```

### Building

```bash
# Build Electron app
bun run build

# Create distributable
bun run dist
```

## Configuration

The backend uses a `.env` file in the `backend/` directory:

```env
# Server
HOST=127.0.0.1
PORT=3900
DEBUG=true

# Compute
DEVICE=auto          # auto, cuda, mps, cpu
FP16=true

# Models
DEFAULT_TTS_ENGINE=omnivoice
DEFAULT_ASR_ENGINE=whisperx

# Auth (for remote admin UI)
REMOTE_API_KEY=your-secure-key

# Paths
DATA_DIR=./data
MODELS_DIR=./models
AUDIO_OUTPUT_DIR=./outputs/audio
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/system/info` | System information |
| `POST /api/generate/clone` | Voice cloning |
| `POST /api/generate/tts` | Standard TTS |
| `POST /api/generate/design` | Voice design |
| `GET /api/profiles` | List voice profiles |
| `POST /api/profiles` | Create voice profile |
| `GET /api/history` | Generation history |
| `GET /api/engines` | Available engines |
| `POST /api/engines/{id}/load` | Load engine |
| `GET /api/audio/{filename}` | Serve generated audio |

## Engine Architecture

Engines run as isolated subprocesses for dependency isolation:

```
backend/engines/
├── omnivoice_subprocess/    # OmniVoice TTS
├── omnivoice_gguf/          # GGUF-quantized OmniVoice
├── xtts_subprocess/         # XTTS v2
├── whisperx_subprocess/     # WhisperX ASR
├── pyannote_subprocess/     # Pyannote diarization
└── _echo/                   # CI regression test
```

Each engine has its own `main.py` entrypoint and communicates via JSON-RPC over stdin/stdout.

## License

- **Electron App**: MIT
- **Backend**: AGPL-3.0
- **OmniVoice Models**: AGPL-3.0

Model weights have their own licenses - review before commercial use.

## Acknowledgments

- Inspired by [VoiceStudio](https://github.com/debpalash/VoiceStudio) architecture
- Built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) models
- Uses [HiFi-GAN](https://github.com/jik876/hifi-gan), [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- UI patterns from [T3 Code](https://github.com/t3dotgg/t3-code) (MIT)
