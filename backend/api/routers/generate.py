"""
Generation endpoints for TTS, voice cloning, voice design
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import asyncio
import tempfile
import shutil
from pathlib import Path
import time

from core.config import settings

router = APIRouter(tags=["generate"])

class GenerateResult(BaseModel):
    audio_path: str
    duration: float
    sample_rate: int
    text: str
    language: str
    engine: str
    seed: Optional[int] = None

# In-memory job tracking (would use Redis in production)
jobs = {}

@router.post("/generate/clone", response_model=GenerateResult)
async def generate_clone(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: str = Form("auto"),
    profile_id: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    ref_audio_path: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    steps: int = Form(30),
    cfg: float = Form(0.7),
    speed: float = Form(1.0),
    t_shift: float = Form(0.0),
    pos_temp: float = Form(0.7),
    class_temp: float = Form(0.7),
    layer_penalty: float = Form(0.0),
    denoise: bool = Form(False),
    postprocess: bool = Form(True),
    duration: Optional[int] = Form(None),
):
    """
    Generate speech with voice cloning
    """
    # Validate input
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty")
    
    if len(text) > settings.max_text_length:
        raise HTTPException(422, f"Text too long (max {settings.max_text_length} chars)")
    
    has_reference = ref_audio or ref_audio_path or profile_id
    if not has_reference:
        raise HTTPException(422, "Reference audio or profile required")
    
    # For now, return a mock result
    # In production, this would call the actual TTS engine
    job_id = str(uuid.uuid4())
    output_filename = f"clone_{job_id}.wav"
    output_path = Path(settings.audio_output_dir) / output_filename
    
    # Create a dummy WAV file for testing
    import wave
    import numpy as np
    
    sample_rate = 22050
    duration_sec = min(len(text) * 0.1, 30)  # Rough estimate
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    # Generate a simple tone
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return GenerateResult(
        audio_path=f"/api/audio/{output_filename}",
        duration=duration_sec,
        sample_rate=sample_rate,
        text=text,
        language=language,
        engine="omnivoice",
        seed=42,
    )

@router.post("/generate/tts", response_model=GenerateResult)
async def generate_tts(
    text: str = Form(...),
    voice: str = Form("default"),
    language: str = Form("en"),
    speed: float = Form(1.0),
    pitch: float = Form(1.0),
):
    """
    Standard TTS generation (no voice cloning)
    """
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty")
    
    job_id = str(uuid.uuid4())
    output_filename = f"tts_{job_id}.wav"
    output_path = Path(settings.audio_output_dir) / output_filename
    
    # Create dummy audio
    import wave
    import numpy as np
    
    sample_rate = 22050
    duration_sec = min(len(text) * 0.1, 30)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return GenerateResult(
        audio_path=f"/api/audio/{output_filename}",
        duration=duration_sec,
        sample_rate=sample_rate,
        text=text,
        language=language,
        engine="omnivoice",
    )

@router.post("/generate/design", response_model=GenerateResult)
async def generate_design(
    text: str = Form(...),
    language: str = Form("en"),
    instruct: str = Form(...),
    speed: float = Form(1.0),
):
    """
    Voice design from natural language description
    """
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty")
    if not instruct.strip():
        raise HTTPException(422, "Voice design instruction required")
    
    job_id = str(uuid.uuid4())
    output_filename = f"design_{job_id}.wav"
    output_path = Path(settings.audio_output_dir) / output_filename
    
    import wave
    import numpy as np
    
    sample_rate = 22050
    duration_sec = min(len(text) * 0.1, 30)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return GenerateResult(
        audio_path=f"/api/audio/{output_filename}",
        duration=duration_sec,
        sample_rate=sample_rate,
        text=text,
        language=language,
        engine="omnivoice",
    )
