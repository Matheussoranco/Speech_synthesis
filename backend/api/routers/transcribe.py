"
Transcription
"

from fastapi import APIRouter

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

@router.post("/generate")
async def transcribe_audio():
    return {"status": "started"}

@router.get("/jobs/{job_id}")
async def get_transcription_job(job_id: str):
    return {"id": job_id, "status": "completed"}
