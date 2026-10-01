"""
Voice profile management
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from core.config import settings

router = APIRouter(tags=["profiles"])

class Profile(BaseModel):
    id: str
    name: str
    ref_audio_path: str
    ref_audio_name: str
    ref_text: str
    instruct: str
    language: str
    created_at: str
    updated_at: str
    starred: bool = False

# In-memory storage (would use database in production)
profiles_db = {}

@router.get("/profiles", response_model=List[Profile])
async def list_profiles():
    return list(profiles_db.values())

@router.post("/profiles", response_model=Profile)
async def create_profile(
    name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_audio_name: str = Form(...),
    ref_text: str = Form(...),
    instruct: str = Form(""),
    language: str = Form("en"),
):
    if not name.strip():
        raise HTTPException(422, "Profile name required")
    
    if not ref_audio.filename:
        raise HTTPException(422, "Reference audio required")
    
    profile_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    # Save reference audio
    audio_dir = Path(settings.audio_output_dir) / "profiles" / profile_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / ref_audio_name
    
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(ref_audio.file, f)
    
    profile = Profile(
        id=profile_id,
        name=name,
        ref_audio_path=str(audio_path),
        ref_audio_name=ref_audio_name,
        ref_text=ref_text,
        instruct=instruct,
        language=language,
        created_at=now,
        updated_at=now,
    )
    
    profiles_db[profile_id] = profile
    return profile

@router.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    return profiles_db[profile_id]

@router.patch("/profiles/{profile_id}", response_model=Profile)
async def update_profile(
    profile_id: str,
    name: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    
    profile = profiles_db[profile_id]
    if name is not None:
        profile.name = name
    if ref_text is not None:
        profile.ref_text = ref_text
    if instruct is not None:
        profile.instruct = instruct
    if language is not None:
        profile.language = language
    profile.updated_at = datetime.utcnow().isoformat()
    
    return profile

@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    
    profile = profiles_db[profile_id]
    # Delete audio file
    try:
        Path(profile.ref_audio_path).unlink(missing_ok=True)
    except Exception:
        pass
    
    del profiles_db[profile_id]
    return {"success": True}

@router.get("/profiles/{profile_id}/audio")
async def get_profile_audio(profile_id: str):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    
    profile = profiles_db[profile_id]
    audio_path = Path(profile.ref_audio_path)
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")
    
    return FileResponse(str(audio_path), media_type="audio/wav", filename=profile.ref_audio_name)
