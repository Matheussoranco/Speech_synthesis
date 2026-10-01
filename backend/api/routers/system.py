"""
System info and health endpoints
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
import platform
import sys
import torch
import psutil
import time

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
    # This would be called from the Electron main process
    # For now, just return success
    return {"success": True, "message": "Restart requested"}
