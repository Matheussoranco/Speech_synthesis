"""
OpenAI-compatible audio API endpoints
Drop-in replacement for OpenAI TTS/STT
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import base64
import io

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# Models
class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)

class TranscriptionRequest(BaseModel):
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json"
    temperature: float = Field(0.0, ge=0.0, le=1.0)

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[dict]] = None
    segments: Optional[List[dict]] = None

class Voice(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

# Available voices (mapped to our engines)
OPENAI_VOICES = {
    "alloy": {"engine": "omnivoice", "voice": "default", "description": "Neutral, balanced"},
    "echo": {"engine": "omnivoice", "voice": "echo", "description": "Warm, expressive"},
    "fable": {"engine": "omnivoice", "voice": "fable", "description": "Storyteller"},
    "onyx": {"engine": "omnivoice", "voice": "onyx", "description": "Deep, authoritative"},
    "nova": {"engine": "omnivoice", "voice": "nova", "description": "Bright, energetic"},
    "shimmer": {"engine": "omnivoice", "voice": "shimmer", "description": "Soft, gentle"},
}

@router.post("/audio/speech")
async def create_speech(
    request: SpeechRequest,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible TTS endpoint
    POST /v1/audio/speech
    """
    # Validate auth (optional - can be disabled)
    # if authorization and not validate_api_key(authorization):
    #     raise HTTPException(401, "Invalid API key")
    
    if not request.input.strip():
        raise HTTPException(422, "Input text cannot be empty")
    
    if len(request.input) > 4096:
        raise HTTPException(422, "Input text too long (max 4096 characters)")
    
    # Map OpenAI voice to our engine
    voice_config = OPENAI_VOICES.get(request.voice, OPENAI_VOICES["alloy"])
    
    # In production, call the actual generation endpoint
    # For now, generate a dummy WAV and convert to requested format
    import wave
    import numpy as np
    
    sample_rate = 24000
    duration = min(len(request.input) * 0.08, 30)
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    wav_buffer.seek(0)
    
    # For non-WAV formats, we'd convert here
    # For simplicity, return WAV regardless of requested format
    media_type = "audio/wav"
    if request.response_format == "mp3":
        media_type = "audio/mpeg"
    elif request.response_format == "opus":
        media_type = "audio/opus"
    elif request.response_format == "flac":
        media_type = "audio/flac"
    
    return StreamingResponse(
        io.BytesIO(wav_buffer.getvalue()),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{request.response_format}"'},
    )

@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible transcription endpoint
    POST /v1/audio/transcriptions
    """
    # Validate file
    if not file.filename:
        raise HTTPException(422, "No file provided")
    
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/flac", "audio/ogg", "audio/webm", "audio/m4a"]
    if file.content_type not in allowed_types:
        raise HTTPException(422, f"Unsupported file type: {file.content_type}")
    
    # In production, call the actual ASR engine
    # For now, return a mock response
    text = "This is a mock transcription of the uploaded audio file."
    
    if response_format == "text":
        return StreamingResponse(
            io.StringIO(text),
            media_type="text/plain",
        )
    elif response_format == "json":
        return TranscriptionResponse(
            text=text,
            language=language or "en",
            duration=5.0,
        )
    elif response_format == "verbose_json":
        return {
            "text": text,
            "language": language or "en",
            "duration": 5.0,
            "words": [],
            "segments": [{"start": 0, "end": 5, "text": text}],
        }
    else:
        return TranscriptionResponse(text=text, language=language or "en")

@router.post("/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible translation endpoint (transcribe + translate to English)
    POST /v1/audio/translations
    """
    # Same as transcription but forces English output
    text = "This is a mock translation of the uploaded audio file to English."
    
    if response_format == "text":
        return StreamingResponse(io.StringIO(text), media_type="text/plain")
    
    return TranscriptionResponse(text=text, language="en", duration=5.0)

@router.get("/audio/voices", response_model=List[Voice])
async def list_voices(authorization: Optional[str] = Header(None)):
    """
    List available voices (OpenAI-compatible extension)
    GET /v1/audio/voices
    """
    return [
        Voice(id=vid, name=vid.capitalize(), info["description"])
        for vid, info in OPENAI_VOICES.items()
    ]

# Health check for OpenAI compat
@router.get("/health")
async def openai_health():
    return {"status": "ok", "service": "voicestudio-openai-compat"}
