"""Engine discovery from actual installed capabilities."""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.system_speech import installed_voices, SpeechUnavailable

router = APIRouter(tags=["engines"])

class EngineInfo(BaseModel):
    id: str
    name: str
    type: str
    status: str
    version: str
    languages: list[str]
    models: list[str]
    device: str
    memory_usage: Optional[int] = None
    capabilities: list[str] = []

class EnginesResponse(BaseModel):
    tts: list[EngineInfo]
    asr: list[EngineInfo]
    diarization: list[EngineInfo]
    translation: list[EngineInfo]
    vocoder: list[EngineInfo]

async def system_engine():
    try:
        voices = await asyncio.to_thread(installed_voices)
    except SpeechUnavailable:
        voices = []
    return EngineInfo(id="system_speech", name="Windows System Speech", type="tts",
        status="available" if voices else "unavailable", version="system",
        languages=sorted({voice["language"] for voice in voices}),
        models=[voice["name"] for voice in voices], device="cpu",
        capabilities=["tts"] if voices else [])

@router.get("/engines", response_model=EnginesResponse)
async def get_engines():
    return EnginesResponse(tts=[await system_engine()], asr=[], diarization=[], translation=[], vocoder=[])

@router.post("/engines/{engine_id}/load", response_model=EngineInfo)
async def load_engine(engine_id: str):
    if engine_id != "system_speech":
        if engine_id not in {"omnivoice", "xtts_v2", "whisperx", "pyannote", "hifigan"}:
            raise HTTPException(404, "Engine not found")
        raise HTTPException(503, "No real adapter is configured for this engine")
    engine = await system_engine()
    if engine.status != "available":
        raise HTTPException(503, "No usable system voice is installed")
    return engine

@router.post("/engines/{engine_id}/unload")
async def unload_engine(engine_id: str):
    if engine_id == "system_speech":
        raise HTTPException(409, "Installed system voices are managed by the operating system")
    raise HTTPException(404, "Engine not loaded")
