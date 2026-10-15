"""Video/audio dubbing: transcribe -> translate -> re-voice each line -> fit to timing -> mux."""
from pathlib import Path

import numpy as np

from . import asr, audio, config, engine, translate
from .languages import code_for as target_code, name_for
from .jobs import Job

MAX_STRETCH = 1.35  # faster than this sounds rushed; regenerate at fixed length instead


def _pick_reference(segments: list[dict], total: float) -> tuple[float, float, str]:
    """Choose a 4-12 s stretch of consecutive speech to clone the original speaker from."""
    best = None
    for i in range(len(segments)):
        start, texts, prev_end = segments[i]["start"], [], segments[i]["start"]
        for seg in segments[i:]:
            if seg["end"] - start > 12 or seg["start"] - prev_end > 1.0:  # too long, or a pause
                break
            prev_end = seg["end"]
            texts.append(seg["text"])
            span = seg["end"] - start
            if span >= 4 and (best is None or span > best[1] - best[0]):
                best = (start, seg["end"], " ".join(texts))
        if best and best[1] - best[0] >= 10:
            break
    if best is None:
        if not segments:
            raise ValueError("Nenhuma fala detectada no arquivo")
        seg = max(segments, key=lambda s: s["end"] - s["start"])
        best = (seg["start"], min(seg["end"], total), seg["text"])
    return best


def run(job: Job, src: Path, work: Path, *, target_language: str, source_language: str = "",
        voice: dict | None = None, original_volume: float = 0.0, num_step: int = 32) -> dict:
    target_language = name_for(target_language)
    work.mkdir(parents=True, exist_ok=True)
    info = audio.probe(src)
    if not info["has_audio"]:
        raise ValueError("O arquivo não tem trilha de áudio")
    total = info["duration"]

    job.update(0.02, "Extraindo áudio")
    source_wav = audio.to_wav(src, work / "source.wav")

    job.update(0.05, "Transcrevendo a fala original")
    transcript = asr.transcribe(source_wav, source_language or None)
    segments = transcript["segments"]
    if not segments:
        raise ValueError("Nenhuma fala detectada no arquivo")
    detected = transcript["language"]

    job.update(0.2, f"Traduzindo {len(segments)} falas para {target_language}")
    if target_code(target_language) == detected:
        translated = segments  # same language: just re-voice
    elif translate.models():
        # Give the GPU to the translation model; ours reload afterwards.
        asr.unload()
        engine.unload()
        lines = translate.translate_lines([s["text"] for s in segments], target_language, source=detected)
        translated = [dict(s, text=t) for s, t in zip(segments, lines)]
    elif target_code(target_language) == "en":
        # Without Ollama, Whisper can still translate any language into English.
        translated = asr.transcribe(source_wav, detected, task="translate")["segments"]
        segments = translated
    else:
        raise translate.TranslationUnavailable(
            "Para traduzir para idiomas além do inglês, instale e abra o Ollama (ollama.com) com algum modelo")

    if voice:
        ref_path, ref_text, cache_key = config.VOICES_DIR / voice["audio_file"], voice["ref_text"], voice["id"]
    else:
        job.update(0.3, "Clonando a voz original")
        start, end, ref_text = _pick_reference(segments, total)
        ref_path, _ = engine.prepare_reference(
            audio.to_wav(source_wav, work / "ref_raw.wav", start=start, duration=end - start), work / "ref.wav")
        cache_key = None

    sr = config.SAMPLE_RATE
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for i, seg in enumerate(translated):
        base = 0.3 + 0.6 * i / len(translated)
        job.update(base, f"Dublando fala {i + 1} de {len(translated)}")
        next_start = translated[i + 1]["start"] if i + 1 < len(translated) else total
        slot = max(next_start - seg["start"], 0.3)

        def prog(p, m, base=base):
            job.update(base + p * 0.6 / len(translated))

        clip = engine.synthesize(seg["text"], language=target_language, ref_audio=ref_path,
                                 ref_text=ref_text, cache_key=cache_key, num_step=num_step, progress=prog)
        clip = audio.trim_silence(clip, sr)
        length = clip.size / sr
        if length > slot * MAX_STRETCH:
            clip = engine.synthesize(seg["text"], language=target_language, ref_audio=ref_path,
                                     ref_text=ref_text, cache_key=cache_key, num_step=num_step,
                                     duration=slot / 1.1, progress=prog)
            clip = audio.trim_silence(clip, sr)
            length = clip.size / sr
        if length > slot:
            clip = audio.time_stretch(clip, length / slot, sr, work / f"seg{i}")
        at = int(seg["start"] * sr)
        clip = clip[: max(track.size - at, 0)]
        track[at:at + clip.size] += clip

    job.update(0.92, "Montando o resultado")
    dubbed_wav = audio.write_wav(track[: int(total * sr)], work / "dubbed.wav")
    srt = work / "dubbed.srt"
    srt.write_text(asr.to_srt(translated), encoding="utf-8")
    files = {"srt": srt}
    if info["has_video"]:
        files["video"] = audio.mux_video(src, dubbed_wav, work / f"{src.stem}_dub.mp4", original_volume)
    else:
        files["audio"] = audio.encode(dubbed_wav, "mp3")
    return {"files": files, "source_language": detected, "lines": translated, "duration": total}
