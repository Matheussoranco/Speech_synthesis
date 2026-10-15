"""
Video dubbing
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/dubbing", tags=["dubbing"])

@router.post("/generate")
async def generate_dubbing():
    raise HTTPException(503, "No real dubbing pipeline is configured")

@router.post("/batch")
async def batch_dubbing():
    raise HTTPException(503, "No real batch dubbing pipeline is configured")
