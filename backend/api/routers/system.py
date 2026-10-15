"""
System info and health endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import platform
import sys
import torch
import psutil
import time
import asyncio
import os
import shutil
from pathlib import Path

from core.config import settings

router = APIRouter(tags=["system"])

class SystemInfo(BaseModel):
    app_version: str
    python_version: str
    torch_version: str
    cuda_version: Optional[str] = None
    device: str
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[int] = None
    gpu_memory_free: Optional[int] = None
    cpu_count: int
    memory_total: int
    memory_free: int
    platform: str
    uptime: float

_start_time = time.time()

@router.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_free = None
    cuda_version = None
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_memory_total = props.total_memory
        gpu_memory_free = torch.cuda.mem_get_info()[0]
    
    mem = psutil.virtual_memory()
    
    return SystemInfo(
        app_version=settings.app_version,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        device=settings.computed_device,
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
        gpu_memory_free=gpu_memory_free,
        cpu_count=psutil.cpu_count(),
        memory_total=mem.total,
        memory_free=mem.available,
        platform=platform.platform(),
        uptime=time.time() - _start_time,
    )

@router.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.app_version}

@router.post("/system/restart-backend")
async def restart_backend():
    raise HTTPException(409, "Restart the owned backend through the desktop application")

def directory_bytes(root: Path) -> int:
    if not root.is_dir() or root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        return 0
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()
                   and not getattr(Path(directory) / name, "is_junction", lambda: False)()]
        for name in files:
            file = Path(directory) / name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except FileNotFoundError:
                    pass
    return total

@router.get("/system/storage")
async def storage_info():
    from api.routers.profiles import profiles_db
    from api.routers.history import history_db
    audio = Path(settings.audio_output_dir)
    cache = Path(settings.cache_dir)
    def measure():
        return {"audio_bytes": directory_bytes(audio), "cache_bytes": directory_bytes(cache),
            "disk_free_bytes": shutil.disk_usage(audio if audio.exists() else audio.parent).free,
            "profile_count": len(profiles_db), "history_count": len(history_db)}
    return await asyncio.to_thread(measure)
