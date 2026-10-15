"""Audio/video helpers built on ffmpeg and soundfile."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

FORMATS = {
    "wav": ("audio/wav", []),
    "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "192k"]),
    "ogg": ("audio/ogg", ["-c:a", "libopus", "-b:a", "96k"]),
    "flac": ("audio/flac", ["-c:a", "flac"]),
}


class MediaError(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(args: list[str]) -> str:
    try:
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", creationflags=_NO_WINDOW)
    except FileNotFoundError as e:
        raise MediaError("ffmpeg não encontrado no PATH") from e
    if done.returncode != 0:
        raise MediaError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "ffmpeg falhou")
    return done.stdout


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                "-of", "json", str(path)])
    info = json.loads(out)
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    return {"duration": float(info.get("format", {}).get("duration") or 0),
            "has_video": "video" in kinds, "has_audio": "audio" in kinds}


def to_wav(src: Path, dst: Path, sr: int = config.SAMPLE_RATE, start: float | None = None,
           duration: float | None = None) -> Path:
    args = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(src)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    _run(args)
    return dst


def read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def write_wav(samples: np.ndarray, path: Path, sr: int = config.SAMPLE_RATE) -> Path:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.99:
        samples = samples * (0.99 / peak)
    sf.write(str(path), samples.astype(np.float32), sr, subtype="PCM_16")
    return path


def encode(wav: Path, fmt: str) -> Path:
    """Convert a WAV into the requested format; returns the new path (or the WAV itself)."""
    if fmt == "wav":
        return wav
    if fmt not in FORMATS:
        raise MediaError(f"Formato não suportado: {fmt}")
    dst = wav.with_suffix(f".{fmt}")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), *FORMATS[fmt][1], str(dst)])
    return dst


def silence(seconds: float, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(max(seconds, 0) * sr), dtype=np.float32)


def trim_silence(samples: np.ndarray, sr: int, threshold: float = 0.01, pad: float = 0.05) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    lo = max(loud[0] - int(pad * sr), 0)
    hi = min(loud[-1] + int(pad * sr), samples.size)
    return samples[lo:hi]


def time_stretch(samples: np.ndarray, factor: float, sr: int, work: Path) -> np.ndarray:
    """Speed audio up (factor > 1) or down without changing pitch, via ffmpeg atempo."""
    if abs(factor - 1.0) < 0.02:
        return samples
    factor = min(max(factor, 0.5), 4.0)
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    src, dst = work.with_suffix(".in.wav"), work.with_suffix(".out.wav")
    write_wav(samples, src, sr)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter:a", ",".join(filters), str(dst)])
    out, _ = read(dst)
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return out


def mux_video(video: Path, audio: Path, dst: Path, original_volume: float = 0.0,
              subtitles: Path | None = None) -> Path:
    """Replace (or mix under) the video's audio track with the dubbed audio."""
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio)]
    if subtitles:
        args += ["-i", str(subtitles)]
    if original_volume > 0:
        args += ["-filter_complex",
                 f"[0:a]volume={original_volume:.2f}[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                 "-map", "0:v", "-map", "[a]"]
    else:
        args += ["-map", "0:v", "-map", "1:a"]
    if subtitles:
        args += ["-map", "2:s", "-c:s", "mov_text"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    _run(args)
    return dst
