"
Engine management
"

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

from core.config import settings

router = APIRouter(tags=["engines"])

class EngineInfo(BaseModel):
    id: str
    name: str
    type: str  # tts, asr, diarization, translation, vocoder
    status: str  # available, loading, error, unavailable
    version: str
    languages: List[str]
    models: List[str]
    device: str
    memory_usage: Optional[int] = None
    capabilities: List[str] = []

class EnginesResponse(BaseModel):
    tts: List[EngineInfo]
    asr: List[EngineInfo]
    diarization: List[EngineInfo]
    translation: List[EngineInfo]
    vocoder: List[EngineInfo]

# Mock engine data
MOCK_ENGINES = {
    "tts": [
        EngineInfo(
            id="omnivoice",
            name="OmniVoice",
            type="tts",
            status="available",
            version="1.0.0",
            languages=["en", "zh", "es", "fr", "de", "ja", "ko", "auto"],
            models=["omnivoice-v1"],
            device=settings.computed_device,
            capabilities=["voice_cloning", "voice_design", "multi_speaker", "streaming"],
        ),
        EngineInfo(
            id="xtts_v2",
            name="XTTS v2",
            type="tts",
            status="unavailable",
            version="2.0.0",
            languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
            models=["xtts_v2"],
            device=settings.computed_device,
            capabilities=["voice_cloning", "multi_speaker"],
        ),
    ],
    "asr": [
        EngineInfo(
            id="whisperx",
            name="WhisperX",
            type="asr",
            status="available",
            version="3.0.0",
            languages=["en", "zh", "es", "fr", "de", "ja", "ko", "auto", "99"],
            models=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
            device=settings.computed_device,
            capabilities=["word_timestamps", "speaker_diarization", "alignment"],
        ),
    ],
    "diarization": [
        EngineInfo(
            id="pyannote",
            name="Pyannote Diarization",
            type="diarization",
            status="unavailable",
            version="3.1.0",
            languages=["*"],
            models=["pyannote/speaker-diarization-3.1"],
            device=settings.computed_device,
            capabilities=["speaker_diarization", "speaker_counting"],
        ),
    ],
    "translation": [],
    "vocoder": [
        EngineInfo(
            id="hifigan",
            name="HiFi-GAN",
            type="vocoder",
            status="available",
            version="1.0.0",
            languages=["*"],
            models=["hifigan_v1", "hifigan_v2", "hifigan_v3"],
            device=settings.computed_device,
            capabilities=["neural_vocoder"],
        ),
    ],
}

@router.get("/engines", response_model=EnginesResponse)
async def get_engines():
    return EnginesResponse(**MOCK_ENGINES)

@router.post("/engines/{engine_id}/load", response_model=EngineInfo)
async def load_engine(engine_id: str):
    # Find engine
    for engines in MOCK_ENGINES.values():
        for engine in engines:
            if engine.id == engine_id:
                engine.status = "available"
                return engine
    raise HTTPException(404, "Engine not found")

@router.post("/engines/{engine_id}/unload")
async def unload_engine(engine_id: str):
    for engines in MOCK_ENGINES.values():
        for engine in engines:
            if engine.id == engine_id:
                engine.status = "unavailable"
                return {"success": True}
    raise HTTPException(404, "Engine not found")
