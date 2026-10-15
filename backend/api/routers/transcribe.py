"""
Transcription
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

@router.post("/generate")
async def transcribe_audio():
    raise HTTPException(503, "No real speech-recognition model is configured")

@router.get("/jobs/{job_id}")
async def get_transcription_job(job_id: str):
    raise HTTPException(404, "Transcription job not found")
