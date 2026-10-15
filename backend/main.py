"""
VoiceStudio Backend - FastAPI application for speech synthesis
Multi-engine architecture with subprocess isolation
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure backend is on sys.path
_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Windows: suppress console windows for subprocesses
if sys.platform == "win32":
    try:
        from core.win_subprocess import install as _install_no_window
        _install_no_window()
    except Exception:
        pass

# Disable Triton on Windows
if sys.platform == "win32":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

# Ensure omnivoice is importable
try:
    from core.omnivoice_path import ensure_omnivoice_importable
    ensure_omnivoice_importable(_backend_dir)
except Exception:
    pass

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.csrf import request_guard
from api.routers import (
    system,
    generate,
    profiles,
    history,
    engines,
    auth,
    audiobook,
    dubbing,
    transcribe,
    archetypes,
    openai_compat,
)
from mcp_shim.server import app as mcp_app
from core.contained_subprocess import secure_backend_drain_fd
from services.model_manager import startup_model_services, shutdown_model_services

# Secure the backend drain fd early
secure_backend_drain_fd()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[backend] Starting VoiceStudio backend v{settings.app_version}")
    await startup_model_services()
    yield
    # Shutdown
    print("[backend] Shutting down...")
    await shutdown_model_services()

app = FastAPI(
    title="VoiceStudio API",
    description="Local-first voice cloning, TTS, dubbing, and audiobook creation",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Registered after CORS so it runs first (outermost).
app.middleware("http")(request_guard)

# Static files for generated audio
audio_dir = Path(settings.audio_output_dir)
audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/audio", StaticFiles(directory=str(audio_dir)), name="audio")

# Include routers
app.include_router(system.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(profiles.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(engines.router, prefix="/api")
app.include_router(audiobook.router, prefix="/api")
app.include_router(dubbing.router, prefix="/api")
app.include_router(transcribe.router, prefix="/api")
app.include_router(archetypes.router, prefix="/api")
app.include_router(openai_compat.router)  # No prefix - OpenAI compat at /v1

# Mount MCP server
app.mount("/mcp", mcp_app)

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[backend] Unhandled exception: {exc}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc) if settings.debug else "Internal server error"},
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "VoiceStudio",
        "version": settings.app_version,
        "description": "Local-first voice cloning, TTS, dubbing, and audiobook creation",
        "docs": "/docs",
        "health": "/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,
    )
