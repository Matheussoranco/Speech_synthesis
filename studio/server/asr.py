"""Speech recognition with OpenAI Whisper (loaded lazily, kept resident)."""
import threading
from pathlib import Path

from . import config

_model = None
_lock = threading.RLock()


def loaded() -> bool:
    return _model is not None


def _load():
    global _model
    if _model is None:
        import whisper

        if config.low_vram():
            from . import engine

            engine.unload(keep_prompts=True)  # small GPU: keep a single model resident

        device = config.resolve_device()
        cached = _cached_checkpoint(config.ASR_MODEL)
        try:
            # Loading the cached file directly avoids whisper reading the whole checkpoint
            # into RAM to verify its hash, which fails on machines short of memory.
            _model = whisper.load_model(cached or config.ASR_MODEL, device=device)
        except Exception:
            if not cached:
                raise
            _model = whisper.load_model(config.ASR_MODEL, device=device)  # re-verify / re-download
    return _model


def _cached_checkpoint(name: str) -> str | None:
    import os

    import whisper

    url = whisper._MODELS.get(name)
    if not url:
        return None
    root = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
    path = os.path.join(root, os.path.basename(url))
    return path if os.path.isfile(path) and os.path.getsize(path) > 0 else None


def unload():
    global _model
    with _lock:
        if _model is None:
            return
        _model = None
        _free()


def _free():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(path: Path, language: str | None = None, task: str = "transcribe") -> dict:
    """Return {'text', 'language', 'segments': [{'start','end','text'}]}."""
    with _lock:
        model = _load()
        result = model.transcribe(
            str(path), language=language or None, task=task,
            fp16=config.resolve_device() == "cuda", condition_on_previous_text=False,
        )
    segments = [{"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"].strip()}
                for s in result.get("segments", []) if s["text"].strip()]
    return {"text": result.get("text", "").strip(), "language": result.get("language", ""),
            "segments": segments}


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_srt(segments: list[dict]) -> str:
    return "\n".join(f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n"
                     for i, s in enumerate(segments, 1))


def to_vtt(segments: list[dict]) -> str:
    return "WEBVTT\n\n" + "\n".join(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n"
                                    for s in segments)
