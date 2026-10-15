"""
Voice archetypes
"""

from fastapi import APIRouter

router = APIRouter(prefix="/archetypes", tags=["archetypes"])

@router.get("/")
async def list_archetypes():
    return []

@router.get("/{archetype_id}")
async def get_archetype(archetype_id: str):
    return {"id": archetype_id}
