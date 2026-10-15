"""Audio API extension backed by installed system voices and real FFmpeg encoding."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from api.routers.generate import generate_tts
from core.config import settings
from services.system_speech import installed_voices, SpeechUnavailable

router = APIRouter(prefix="/v1", tags=["openai-compat"])

class SpeechRequest(BaseModel):
    model: str = "system_speech"
    input: str
    voice: str = "default"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(1.0, ge=0.5, le=2.0)

class Voice(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

FORMATS = {
    "mp3": ("libmp3lame", "mp3", "audio/mpeg"),
    "opus": ("libopus", "opus", "audio/ogg"),
    "aac": ("aac", "adts", "audio/aac"),
    "flac": ("flac", "flac", "audio/flac"),
    "pcm": ("pcm_s16le", "s16le", "application/octet-stream"),
}

def _encode(source: Path, target: Path, response_format: str):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise HTTPException(503, "FFmpeg must be installed to encode this audio format")
    codec, container, _ = FORMATS[response_format]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source), "-ac", "1", "-ar", "24000", "-c:a", codec,
            "-f", container, str(target)], check=True, capture_output=True,
            timeout=settings.cpu_generate_timeout_s, creationflags=flags)
    except (OSError, subprocess.SubprocessError) as error:
        raise HTTPException(503, "Audio encoding failed") from error

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, authorization: Optional[str] = Header(None)):
    if request.model != "system_speech":
        raise HTTPException(404, "Only the installed system_speech model is supported")
    if len(request.input) > 4096:
        raise HTTPException(422, "Input text too long (max 4096 characters)")
    if request.response_format != "wav" and not shutil.which("ffmpeg"):
        raise HTTPException(503, "FFmpeg must be installed to encode this audio format")
    result = await generate_tts(text=request.input, voice=request.voice, language="auto",
        speed=request.speed, pitch=1.0)
    source = Path(settings.audio_output_dir) / Path(result.audio_path).name
    if request.response_format == "wav":
        return FileResponse(source, media_type="audio/wav", filename="speech.wav")
    descriptor, filename = tempfile.mkstemp(prefix="speech-", suffix="." + request.response_format,
        dir=settings.audio_output_dir)
    os.close(descriptor)
    target = Path(filename)
    try:
        await asyncio.to_thread(_encode, source, target, request.response_format)
        return FileResponse(target, media_type=FORMATS[request.response_format][2],
            filename="speech." + request.response_format, background=BackgroundTask(target.unlink, missing_ok=True))
    except BaseException:
        target.unlink(missing_ok=True)
        raise

@router.post("/audio/transcriptions")
async def create_transcription(file: UploadFile = File(...), model: str = Form("whisper-1"),
    language: Optional[str] = Form(None), prompt: Optional[str] = Form(None),
    response_format: str = Form("json"), temperature: float = Form(0.0),
    authorization: Optional[str] = Header(None)):
    raise HTTPException(503, "No real speech-recognition model is configured")

@router.post("/audio/translations")
async def create_translation(file: UploadFile = File(...), model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None), response_format: str = Form("json"),
    temperature: float = Form(0.0), authorization: Optional[str] = Header(None)):
    raise HTTPException(503, "No real speech translation model is configured")

@router.get("/audio/voices", response_model=list[Voice])
async def list_voices(authorization: Optional[str] = Header(None)):
    try:
        voices = await asyncio.to_thread(installed_voices)
    except SpeechUnavailable:
        voices = []
    return [Voice(id=voice["name"], name=voice["name"], description=voice["language"]) for voice in voices]

@router.get("/health")
async def openai_health():
    return {"status": "ok", "service": "voicestudio-audio-api"}
