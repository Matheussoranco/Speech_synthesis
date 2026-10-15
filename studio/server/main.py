"""HTTP API and web UI for the local speech studio."""
import asyncio
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import asr, audio, config, dubbing, engine, jobs, store, translate
from .languages import as_list as language_list

app = FastAPI(title="Speech Studio", version="0.1.0", docs_url="/api/docs", redoc_url=None)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Reject requests whose Host header is not local (blocks DNS-rebinding from web pages)
    # unless the server was explicitly bound to another interface.
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if config.HOST in ("127.0.0.1", "localhost") and host not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "Host não permitido"}, status_code=403)
    response = await call_next(request)
    if not request.url.path.startswith("/files/"):
        response.headers["Cache-Control"] = "no-cache"  # UI updates show up without a hard reload
    return response


app.mount("/files/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
app.mount("/files/voices", StaticFiles(directory=config.VOICES_DIR), name="voices")


# ── helpers ──────────────────────────────────────────────────────────────


def _url(path: Path) -> str:
    path = Path(path).resolve()
    for base, prefix in ((config.OUTPUT_DIR, "/files/outputs/"), (config.VOICES_DIR, "/files/voices/")):
        if path.is_relative_to(base):
            return prefix + path.relative_to(base).as_posix()
    raise ValueError(f"{path} is outside the data directory")


def _job_dir(kind: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = config.OUTPUT_DIR / f"{stamp}-{kind}-{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True)
    return d


async def _save_upload(upload: UploadFile, folder: Path, stem: str = "upload") -> Path:
    suffix = Path(upload.filename or "").suffix.lower()[:10] or ".bin"
    dst = folder / f"{stem}{suffix}"
    size = 0
    with dst.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                fh.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"Arquivo maior que {config.MAX_UPLOAD_BYTES // 2**20} MB")
            fh.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(422, "Arquivo vazio")
    return dst


def _check_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(422, "Digite um texto")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(422, f"Texto longo demais (máximo {config.MAX_TEXT_CHARS} caracteres)")
    return text


def _voice_or_404(voice_id: str) -> dict:
    voice = store.get_voice(voice_id)
    if not voice:
        raise HTTPException(404, "Voz não encontrada")
    return voice


def _validate_instruct(instruct: str) -> str:
    items = [i.strip().lower() for i in (instruct or "").split(",") if i.strip()]
    valid = {o for opts in engine.DESIGN_OPTIONS.values() for o in opts}
    bad = [i for i in items if i not in valid]
    if bad:
        raise HTTPException(422, f"Atributos de voz desconhecidos: {', '.join(bad)}")
    return ", ".join(items)


# ── status ───────────────────────────────────────────────────────────────


@app.get("/api/status")
def status():
    import torch

    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "vram_total_mb": total // 2**20,
               "vram_free_mb": free // 2**20}
    return {
        "device": config.resolve_device(), "gpu": gpu,
        "tts_loaded": engine.loaded(), "asr_loaded": asr.loaded(),
        "ffmpeg": audio.have_ffmpeg(), "translation_models": translate.models(),
        "languages": language_list(), "design_options": engine.DESIGN_OPTIONS,
        "formats": list(audio.FORMATS), "jobs": jobs.active(),
    }


@app.post("/api/unload")
def unload():
    engine.unload()
    asr.unload()
    return {"ok": True}


# ── jobs ─────────────────────────────────────────────────────────────────


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Tarefa não encontrada")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return {"cancelled": jobs.cancel(job_id)}


# ── generation (TTS, voice design, cloning, long-form) ───────────────────


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    mode: str = Form("auto"),  # auto | design | voice | clone
    voice_id: str = Form(""),
    instruct: str = Form(""),
    ref_audio: UploadFile | None = File(None),
    ref_text: str = Form(""),
    language: str = Form(""),
    speed: float = Form(1.0),
    num_step: int = Form(32),
    seed: int = Form(-1),
    format: str = Form("wav"),
):
    text = _check_text(text)
    if format not in audio.FORMATS:
        raise HTTPException(422, "Formato inválido")
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(422, "Velocidade deve estar entre 0.5 e 2.0")
    num_step = max(8, min(num_step, 64))
    out_dir = _job_dir("tts")
    kwargs: dict = dict(language=language, speed=speed, num_step=num_step, seed=seed)
    meta: dict = {"text": text, "mode": mode, "language": language, "speed": speed}

    if mode == "design":
        kwargs["instruct"] = meta["instruct"] = _validate_instruct(instruct)
    elif mode == "voice":
        voice = _voice_or_404(voice_id)
        kwargs.update(ref_audio=config.VOICES_DIR / voice["audio_file"], ref_text=voice["ref_text"],
                      cache_key=voice["id"])
        meta["voice"] = voice["name"]
    elif mode == "clone":
        if not ref_audio:
            raise HTTPException(422, "Envie um áudio de referência")
        raw = await _save_upload(ref_audio, out_dir, "reference_upload")
        kwargs["ref_text"] = ref_text.strip()
        kwargs["ref_audio"] = raw  # normalised inside the job
    elif mode != "auto":
        raise HTTPException(422, "Modo inválido")

    def work(job: jobs.Job) -> dict:
        if mode == "clone":
            kwargs["ref_audio"], _ = engine.prepare_reference(kwargs["ref_audio"], out_dir / "reference.wav")
        samples = engine.synthesize(text, progress=lambda p, m: job.update(p * 0.95, m), **kwargs)
        wav = audio.write_wav(samples, out_dir / "speech.wav")
        final = audio.encode(wav, format)
        duration = samples.size / config.SAMPLE_RATE
        files = {"audio": _url(final)}
        if final != wav:
            files["wav"] = _url(wav)
        item = store.add_history(mode, text, files, dict(meta, duration=duration))
        return {"audio_url": files["audio"], "duration": duration, "history_id": item["id"]}

    titles = {"auto": "Fala", "design": "Voz desenhada", "voice": "Voz salva", "clone": "Clonagem"}
    return jobs.submit("generate", titles[mode], work).as_dict()


# ── saved voices ─────────────────────────────────────────────────────────


@app.get("/api/voices")
def voices():
    return [dict(v, audio_url=_url(config.VOICES_DIR / v["audio_file"])) for v in store.list_voices()]


def _make_voice(job: jobs.Job, src: Path, name: str, ref_text: str, language: str,
                source: str, instruct: str = "") -> dict:
    job.update(0.1, "Preparando a referência")
    file_name = f"{uuid.uuid4().hex[:12]}.wav"
    dst, seconds = engine.prepare_reference(src, config.VOICES_DIR / file_name)
    if not ref_text:
        job.update(0.4, "Transcrevendo a referência")
        ref_text = engine.transcribe_reference(dst, language)
    voice = store.add_voice(name, file_name, ref_text, language, source, instruct)
    return dict(voice, seconds=seconds)


@app.post("/api/voices")
async def create_voice(name: str = Form(...), audio_file: UploadFile = File(...),
                       ref_text: str = Form(""), language: str = Form("")):
    name = name.strip()[:80] or "Minha voz"
    tmp = await _save_upload(audio_file, config.UPLOAD_DIR, uuid.uuid4().hex)

    def work(job):
        try:
            return _make_voice(job, tmp, name, ref_text.strip(), language, "upload")
        finally:
            tmp.unlink(missing_ok=True)

    return jobs.submit("voice", f"Salvar voz {name}", work).as_dict()


@app.post("/api/voices/from-history")
def voice_from_history(history_id: str = Form(...), name: str = Form(...)):
    item = store.get_history(history_id)
    if not item:
        raise HTTPException(404, "Item do histórico não encontrado")
    src = config.OUTPUT_DIR / Path(item["files"].get("wav") or item["files"]["audio"]).relative_to("/files/outputs")
    text = item["meta"].get("text", "")
    # The transcript only matches when the whole clip fits in the reference window.
    ref_text = text if item["meta"].get("duration", 99) <= config.REF_MAX_SECONDS else ""
    name = name.strip()[:80] or "Voz"
    return jobs.submit("voice", f"Salvar voz {name}", lambda job: _make_voice(
        job, src, name, ref_text, item["meta"].get("language", ""), item["kind"],
        item["meta"].get("instruct", ""))).as_dict()


class VoicePatch(BaseModel):
    name: str | None = None
    ref_text: str | None = None
    language: str | None = None


@app.patch("/api/voices/{voice_id}")
def update_voice(voice_id: str, patch: VoicePatch):
    _voice_or_404(voice_id)
    engine.forget_voice(voice_id)
    return store.update_voice(voice_id, **patch.model_dump())


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice = _voice_or_404(voice_id)
    store.delete_voice(voice_id)
    engine.forget_voice(voice_id)
    (config.VOICES_DIR / voice["audio_file"]).unlink(missing_ok=True)
    return {"ok": True}


# ── transcription ────────────────────────────────────────────────────────


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(""), task: str = Form("transcribe")):
    if task not in ("transcribe", "translate"):
        raise HTTPException(422, "Tarefa inválida")
    out_dir = _job_dir("transcript")
    src = await _save_upload(file, out_dir, "source")
    title = file.filename or "Transcrição"

    def work(job):
        job.update(0.05, "Carregando o reconhecimento de fala" if not asr.loaded() else "Transcrevendo")
        result = asr.transcribe(src, language or None, task)
        job.update(0.95, "Salvando")
        (out_dir / "transcript.txt").write_text(result["text"], encoding="utf-8")
        (out_dir / "transcript.srt").write_text(asr.to_srt(result["segments"]), encoding="utf-8")
        (out_dir / "transcript.vtt").write_text(asr.to_vtt(result["segments"]), encoding="utf-8")
        src.unlink(missing_ok=True)
        files = {k: _url(out_dir / f"transcript.{k}") for k in ("txt", "srt", "vtt")}
        item = store.add_history("transcribe", title, files,
                                 {"text": result["text"], "language": result["language"]})
        return dict(result, files=files, history_id=item["id"])

    return jobs.submit("transcribe", title, work).as_dict()


# ── dubbing ──────────────────────────────────────────────────────────────


@app.post("/api/dub")
async def dub(file: UploadFile = File(...), target_language: str = Form(...), source_language: str = Form(""),
              voice_id: str = Form(""), original_volume: float = Form(0.0), num_step: int = Form(24)):
    if not audio.have_ffmpeg():
        raise HTTPException(503, "Dublagem precisa do ffmpeg instalado")
    voice = _voice_or_404(voice_id) if voice_id else None
    out_dir = _job_dir("dub")
    src = await _save_upload(file, out_dir, "source")
    title = file.filename or "Dublagem"

    def work(job):
        result = dubbing.run(job, src, out_dir, target_language=target_language, source_language=source_language,
                             voice=voice, original_volume=max(0.0, min(original_volume, 1.0)),
                             num_step=max(8, min(num_step, 64)))
        files = {k: _url(p) for k, p in result["files"].items()}
        item = store.add_history("dub", title, files, {"target_language": target_language,
                                                       "source_language": result["source_language"],
                                                       "duration": result["duration"]})
        return {"files": files, "lines": result["lines"], "history_id": item["id"]}

    return jobs.submit("dub", title, work).as_dict()


# ── history ──────────────────────────────────────────────────────────────


@app.get("/api/history")
def history(limit: int = 200):
    return store.list_history(limit)


@app.delete("/api/history/{item_id}")
def delete_history(item_id: str):
    item = store.delete_history(item_id)
    if not item:
        raise HTTPException(404, "Item não encontrado")
    folders = {(config.OUTPUT_DIR / Path(u).relative_to("/files/outputs")).parent
               for u in item["files"].values() if u.startswith("/files/outputs/")}
    for folder in folders:
        if folder.resolve().parent == config.OUTPUT_DIR:
            shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True}


# ── OpenAI-compatible endpoints (for scripts and other apps) ─────────────


class SpeechRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = "auto"
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str = ""
    language: str = ""


async def _wait(job: jobs.Job) -> dict:
    while job.status in ("queued", "running"):
        await asyncio.sleep(0.2)
    if job.status != "done":
        raise HTTPException(500, job.error or "Falhou")
    return job.result


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest):
    text = _check_text(req.input)
    fmt = {"pcm": "wav", "aac": "mp3", "opus": "ogg"}.get(req.response_format, req.response_format)
    if fmt not in audio.FORMATS:
        raise HTTPException(422, "response_format inválido")
    voice = store.find_voice(req.voice) if req.voice else None
    kwargs: dict = dict(language=req.language, speed=max(0.5, min(req.speed, 2.0)))
    if voice:
        kwargs.update(ref_audio=config.VOICES_DIR / voice["audio_file"], ref_text=voice["ref_text"],
                      cache_key=voice["id"])
    elif req.instructions:
        kwargs["instruct"] = _validate_instruct(req.instructions)
    out_dir = _job_dir("api")

    def work(job):
        samples = engine.synthesize(text, progress=lambda p, m: job.update(p), **kwargs)
        return {"path": str(audio.encode(audio.write_wav(samples, out_dir / "speech.wav"), fmt))}

    result = await _wait(jobs.submit("generate", "API", work))
    data = Path(result["path"]).read_bytes()
    shutil.rmtree(out_dir, ignore_errors=True)
    return Response(data, media_type=audio.FORMATS[fmt][0])


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form("whisper-1"),
                                language: str = Form(""), response_format: str = Form("json")):
    src = await _save_upload(file, config.UPLOAD_DIR, uuid.uuid4().hex)

    def work(job):
        try:
            return asr.transcribe(src, language or None)
        finally:
            src.unlink(missing_ok=True)

    result = await _wait(jobs.submit("transcribe", "API", work))
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "srt":
        return PlainTextResponse(asr.to_srt(result["segments"]))
    if response_format == "vtt":
        return PlainTextResponse(asr.to_vtt(result["segments"]))
    if response_format == "verbose_json":
        return result
    return {"text": result["text"]}


@app.get("/v1/models")
def openai_models():
    return {"object": "list", "data": [{"id": "tts-1", "object": "model"}, {"id": "whisper-1", "object": "model"}]}


# ── web UI ───────────────────────────────────────────────────────────────


@app.get("/")
def index():
    return FileResponse(config.UI_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.UI_DIR), name="ui")
