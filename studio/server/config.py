"""Runtime settings, read from environment variables (prefix STUDIO_)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"STUDIO_{name}", default)


HOST = _env("HOST", "127.0.0.1")
PORT = int(_env("PORT", "7860"))

DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data"))).resolve()
VOICES_DIR = DATA_DIR / "voices"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "studio.db"
UI_DIR = ROOT / "ui"

TTS_MODEL = _env("TTS_MODEL", "k2-fsa/OmniVoice")
ASR_MODEL = _env("ASR_MODEL", "turbo")  # any openai-whisper size: tiny, base, small, medium, turbo, large-v3
DEVICE = _env("DEVICE", "auto")  # auto, cuda, cpu

OLLAMA_URL = _env("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "")  # empty = first installed model

MAX_UPLOAD_BYTES = int(_env("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
MAX_TEXT_CHARS = int(_env("MAX_TEXT_CHARS", "200000"))
REF_MAX_SECONDS = float(_env("REF_MAX_SECONDS", "15"))
SAMPLE_RATE = 24000

for _d in (VOICES_DIR, OUTPUT_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def low_vram() -> bool:
    """True on GPUs too small to hold the voice and speech-recognition models together."""
    if _env("LOW_VRAM", "auto") != "auto":
        return _env("LOW_VRAM", "auto") in ("1", "true", "yes")
    if resolve_device() != "cuda":
        return False
    import torch

    return torch.cuda.get_device_properties(0).total_memory < 10 * 2**30
