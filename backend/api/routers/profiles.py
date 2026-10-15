"""
Voice profile management
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import mimetypes
import errno
from pathlib import Path
from datetime import datetime, timezone

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

def validate_audio_name(name: str) -> str:
    # Enforce a portable leaf name, including Windows ADS and device aliases.
    if (not name or name in {".", ".."} or name != name.rstrip(" .")
        or any(ord(char) < 32 or ord(char) == 127 or char in '/\\:<>"|?*' for char in name)
        or name.split(".", 1)[0].rstrip(" ").upper() in
        {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
         *[f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³"]}):
        raise HTTPException(422, "Reference audio name must be a portable file name")
    return name

def is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())

def profile_root() -> Path:
    root = Path(settings.audio_output_dir).resolve() / "profiles"
    if is_link(root):
        raise HTTPException(409, "Unsafe profile storage directory")
    return root

def stored_audio_path(profile: Profile) -> Path:
    try:
        if str(uuid.UUID(profile.id)) != profile.id:
            raise ValueError("Invalid profile identifier")
        name = validate_audio_name(profile.ref_audio_name)
        root = profile_root()
        directory = root / profile.id
        expected = directory / name
        if (is_link(directory) or is_link(expected)
            or Path(profile.ref_audio_path).absolute() != expected
            or expected.resolve() != expected):
            raise ValueError("Unsafe stored path")
        return expected
    except (ValueError, OSError, HTTPException) as error:
        raise HTTPException(409, "Unsafe stored profile audio path") from error

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
    validate_audio_name(ref_audio_name)
    
    profile_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    # Save reference audio
    root = profile_root()
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / profile_id
    audio_dir.mkdir(exist_ok=False)
    audio_path = audio_dir / ref_audio_name
    
    try:
        with audio_path.open("xb") as output:
            size = 0
            while chunk := await ref_audio.read(64 * 1024):
                size += len(chunk)
                if size > settings.max_profile_upload_bytes:
                    raise HTTPException(413, "Reference audio exceeds the upload limit")
                output.write(chunk)
            if not size:
                raise HTTPException(422, "Reference audio cannot be empty")
    except BaseException:
        audio_path.unlink(missing_ok=True)
        audio_dir.rmdir()
        raise
    
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
        if not name.strip():
            raise HTTPException(422, "Profile name required")
        profile.name = name
    if ref_text is not None:
        profile.ref_text = ref_text
    if instruct is not None:
        profile.instruct = instruct
    if language is not None:
        profile.language = language
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    
    return profile

@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    
    profile = profiles_db[profile_id]
    # Delete audio file
    audio_path = stored_audio_path(profile)
    audio_path.unlink(missing_ok=True)
    if audio_path.parent.exists():
        try:
            audio_path.parent.rmdir()
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise
    
    del profiles_db[profile_id]
    return {"success": True}

@router.get("/profiles/{profile_id}/audio")
async def get_profile_audio(profile_id: str):
    if profile_id not in profiles_db:
        raise HTTPException(404, "Profile not found")
    
    profile = profiles_db[profile_id]
    audio_path = stored_audio_path(profile)
    if not audio_path.is_file():
        raise HTTPException(404, "Audio file not found")
    
    media_type = {".webm": "audio/webm", ".m4a": "audio/mp4", ".ogg": "audio/ogg"}.get(audio_path.suffix.lower())
    return FileResponse(str(audio_path), media_type=media_type or mimetypes.guess_type(profile.ref_audio_name)[0] or "application/octet-stream", filename=profile.ref_audio_name)
