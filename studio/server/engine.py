"""Speech synthesis engine (OmniVoice): auto voice, voice design and voice cloning.

Long texts are split into sentence chunks so progress can be reported and
memory stays bounded. When no reference voice is given, the first chunk's
audio becomes the reference for the remaining chunks so the voice stays
consistent across the whole text.
"""
import re
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from . import asr, audio, config

_model = None
_lock = threading.RLock()
_prompt_cache: dict[str, object] = {}

CHUNK_CHARS = 280
BATCH = 4  # chunks generated together once the voice is fixed
PAUSE_SENTENCE = 0.25
PAUSE_PARAGRAPH = 0.6

# Voice-design attributes the model understands (English tags).
DESIGN_OPTIONS = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
    "accent": ["american accent", "british accent", "australian accent", "canadian accent",
               "indian accent", "chinese accent", "korean accent", "japanese accent",
               "portuguese accent", "russian accent"],
}


def loaded() -> bool:
    return _model is not None


def _load():
    global _model
    if _model is None:
        import torch
        from omnivoice import OmniVoice

        device = config.resolve_device()
        if config.low_vram():
            asr.unload()  # small GPU: keep a single model resident
        if device == "cuda":
            _model = OmniVoice.from_pretrained(config.TTS_MODEL, device_map="cuda:0", dtype=torch.float16)
        else:
            _model = OmniVoice.from_pretrained(config.TTS_MODEL, device_map="cpu", dtype=torch.float32)
    return _model


def unload(keep_prompts: bool = False):
    global _model
    with _lock:
        if _model is None:
            return
        _model = None
        if not keep_prompts:
            _prompt_cache.clear()
        asr._free()


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[str, float]]:
    """Split text into (chunk, pause_after_seconds) pieces on paragraph/sentence boundaries."""
    pieces: list[tuple[str, float]] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?;:。！？…])\s+", para)
        current = ""
        for sentence in sentences:
            while len(sentence) > max_chars:  # very long sentence: break on commas/spaces
                cut = max(sentence.rfind(", ", 0, max_chars), sentence.rfind(" ", 0, max_chars))
                cut = cut if cut > max_chars // 3 else max_chars
                head, sentence = sentence[:cut + 1].strip(), sentence[cut + 1:].strip()
                if current:
                    pieces.append((current, PAUSE_SENTENCE))
                    current = ""
                pieces.append((head, 0.12))
            if current and len(current) + len(sentence) + 1 > max_chars:
                pieces.append((current, PAUSE_SENTENCE))
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append((current, PAUSE_SENTENCE))
        if pieces:
            pieces[-1] = (pieces[-1][0], PAUSE_PARAGRAPH)
    return pieces


def prepare_reference(src: Path, dst: Path) -> tuple[Path, float]:
    """Normalise a reference clip to mono 24 kHz WAV, trimmed to REF_MAX_SECONDS."""
    audio.to_wav(src, dst)
    samples, sr = audio.read(dst)
    samples = audio.trim_silence(samples, sr)
    limit = int(config.REF_MAX_SECONDS * sr)
    if samples.size > limit:
        # Cut at the quietest point in the last 3 seconds before the limit.
        window = samples[limit - 3 * sr:limit]
        frame = int(0.05 * sr)
        energy = [float(np.abs(window[i:i + frame]).mean()) for i in range(0, window.size - frame, frame)]
        cut = limit - 3 * sr + int(np.argmin(energy)) * frame if energy else limit
        samples = samples[:cut]
    if samples.size < sr * 1.0:
        raise ValueError("A referência precisa ter pelo menos 1 segundo de fala")
    audio.write_wav(samples, dst, sr)
    return dst, samples.size / sr


def transcribe_reference(path: Path, language: str = "") -> str:
    return asr.transcribe(path, language or None)["text"]


def _clone_prompt(model, ref_audio: Path, ref_text: str, cache_key: str | None):
    if cache_key and cache_key in _prompt_cache:
        return _prompt_cache[cache_key]
    prompt = model.create_voice_clone_prompt(str(ref_audio), ref_text=ref_text or None)
    if cache_key:
        _prompt_cache[cache_key] = prompt
    return prompt


def forget_voice(voice_id: str):
    _prompt_cache.pop(voice_id, None)


def synthesize(
    text: str,
    *,
    language: str = "",
    instruct: str = "",
    ref_audio: Path | None = None,
    ref_text: str = "",
    cache_key: str | None = None,
    speed: float = 1.0,
    duration: float | None = None,
    num_step: int = 32,
    seed: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """Synthesize `text` and return mono float32 samples at config.SAMPLE_RATE."""
    import torch

    progress = progress or (lambda p, m: None)
    chunks = [(text.strip(), 0.0)] if duration else split_text(text)
    if not chunks:
        raise ValueError("Texto vazio")
    with _lock:
        if ref_audio is not None and not ref_text and not (cache_key and cache_key in _prompt_cache):
            progress(0.0, "Transcrevendo a referência")
            ref_text = transcribe_reference(ref_audio, language)
        if not loaded():
            progress(0.0, "Carregando modelo de voz (a primeira vez baixa ~2,5 GB)")
        model = _load()
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
        prompt = _clone_prompt(model, ref_audio, ref_text, cache_key) if ref_audio is not None else None

        kwargs = dict(language=language or None, num_step=num_step, speed=speed, duration=duration)
        clips: list[np.ndarray] = []
        i = 0
        while i < len(chunks):
            progress(i / len(chunks), f"Gerando trecho {i + 1} de {len(chunks)}")
            if prompt is None:
                out = model.generate(text=chunks[i][0], instruct=instruct or None, **kwargs)
                batch = [chunks[i][0]]
            else:
                batch = [c for c, _ in chunks[i:i + BATCH]]
                out = model.generate(text=batch, voice_clone_prompt=[prompt] * len(batch), **kwargs)
            clips.extend(np.asarray(o, dtype=np.float32).reshape(-1) for o in out)
            if prompt is None and len(chunks) > 1:
                # Lock the voice chosen for the first chunk for the rest of the text.
                prompt = model.create_voice_clone_prompt((torch.from_numpy(clips[0]), config.SAMPLE_RATE),
                                                         ref_text=chunks[0][0])
            i += len(batch)
        progress(1.0, "Finalizando")
    parts: list[np.ndarray] = []
    for n, (clip, (_, pause)) in enumerate(zip(clips, chunks)):
        parts.append(clip)
        if n < len(clips) - 1:
            parts.append(audio.silence(pause))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
