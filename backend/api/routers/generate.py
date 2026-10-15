"""Speech generation endpoints. Unsupported model capabilities fail explicitly."""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import asyncio
import uuid
import wave
from core.config import settings
from services.system_speech import synthesize, SpeechUnavailable
from api.routers.history import add_history_entry

router = APIRouter(tags=["generate"])

class GenerateResult(BaseModel):
    audio_path: str
    duration: float
    sample_rate: int
    text: str
    language: str
    engine: str
    seed: Optional[int] = None

def validate_text(text: str):
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty")
    if len(text) > settings.max_text_length:
        raise HTTPException(422, f"Text too long (max {settings.max_text_length} chars)")

@router.post("/generate/clone", response_model=GenerateResult)
async def generate_clone(
    background_tasks: BackgroundTasks,
    text: str = Form(...), language: str = Form("auto"),
    profile_id: Optional[str] = Form(None), ref_audio: Optional[UploadFile] = File(None),
    ref_audio_path: Optional[str] = Form(None), instruct: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None), steps: int = Form(30), cfg: float = Form(0.7),
    speed: float = Form(1.0), t_shift: float = Form(0.0), pos_temp: float = Form(0.7),
    class_temp: float = Form(0.7), layer_penalty: float = Form(0.0),
    denoise: bool = Form(False), postprocess: bool = Form(True), duration: Optional[int] = Form(None),
):
    validate_text(text)
    if not (ref_audio or ref_audio_path or profile_id):
        raise HTTPException(422, "Reference audio or profile required")
    raise HTTPException(503, "No real voice-cloning model is configured")

@router.post("/generate/tts", response_model=GenerateResult)
async def generate_tts(
    text: str = Form(...), voice: str = Form("default"), language: str = Form("en"),
    speed: float = Form(1.0, ge=0.5, le=2.0), pitch: float = Form(1.0),
):
    validate_text(text)
    if pitch != 1.0:
        raise HTTPException(422, "Installed system voices do not support the requested pitch control")
    output_dir = Path(settings.audio_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tts_{uuid.uuid4()}.wav"
    try:
        selected = await asyncio.to_thread(synthesize, text, output_path, voice=voice,
            language=language, speed=speed, timeout=settings.cpu_generate_timeout_s)
        with wave.open(str(output_path), "rb") as audio:
            sample_rate = audio.getframerate()
            duration = audio.getnframes() / sample_rate
            if not 0 < duration <= settings.max_audio_duration:
                raise SpeechUnavailable("Speech output duration is invalid or exceeds the configured limit")
        result = GenerateResult(audio_path=f"/api/audio/{output_path.name}", duration=duration,
            sample_rate=sample_rate, text=text, language=selected["language"], engine="system_speech")
        add_history_entry(type="tts", text=text, audio_path=result.audio_path, duration=duration,
            language=result.language, engine=result.engine, metadata={"voice": selected["name"]})
        return result
    except (SpeechUnavailable, wave.Error, OSError) as error:
        output_path.unlink(missing_ok=True)
        raise HTTPException(503, str(error)) from error

@router.post("/generate/design", response_model=GenerateResult)
async def generate_design(text: str = Form(...), language: str = Form("en"),
    instruct: str = Form(...), speed: float = Form(1.0)):
    validate_text(text)
    if not instruct.strip():
        raise HTTPException(422, "Voice design instruction required")
    raise HTTPException(503, "No real voice-design model is configured")
