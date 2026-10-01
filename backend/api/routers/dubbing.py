"
Video dubbing
"

from fastapi import APIRouter

router = APIRouter(prefix="/dubbing", tags=["dubbing"])

@router.post("/generate")
async def generate_dubbing():
    return {"status": "started"}

@router.post("/batch")
async def batch_dubbing():
    return {"status": "started"}
