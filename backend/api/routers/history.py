"
Generation history
"

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

router = APIRouter(tags=["history"])

class HistoryItem(BaseModel):
    id: str
    type: str  # clone, tts, design, dubbing, transcribe
    text: str
    audio_path: str
    duration: float
    language: str
    engine: str
    metadata: Dict[str, Any] = {}
    created_at: str
    starred: bool = False

# In-memory storage
history_db = []

@router.get("/history", response_model=List[HistoryItem])
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return history_db[offset:offset + limit]

@router.delete("/history")
async def clear_history():
    global history_db
    history_db = []
    return {"success": True, "cleared": True}

@router.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    global history_db
    history_db = [h for h in history_db if h.id != item_id]
    return {"success": True}

@router.patch("/history/{item_id}/star", response_model=HistoryItem)
async def set_history_starred(item_id: str, starred: bool):
    for item in history_db:
        if item.id == item_id:
            item.starred = starred
            return item
    raise HTTPException(404, "History item not found")

# Helper to add history entries (called from generate endpoints)
def add_history_entry(
    type: str,
    text: str,
    audio_path: str,
    duration: float,
    language: str,
    engine: str,
    metadata: Dict[str, Any] = {},
):
    item = HistoryItem(
        id=str(uuid.uuid4()),
        type=type,
        text=text,
        audio_path=audio_path,
        duration=duration,
        language=language,
        engine=engine,
        metadata=metadata,
        created_at=datetime.utcnow().isoformat(),
    )
    history_db.insert(0, item)
    # Keep only last 1000 entries
    if len(history_db) > 1000:
        history_db.pop()
    return item
