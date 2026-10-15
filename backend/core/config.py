"""
Core configuration for VoiceStudio backend
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # App
    app_name: str = "VoiceStudio"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Server
    host: str = "127.0.0.1"
    port: int = 3900
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3901", "http://localhost:3902", "voicestudio://voicestudio"]
    
    # Paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    audio_output_dir: Path = Path("./outputs/audio")
    cache_dir: Path = Path("./cache")
    
    # Model settings
    default_tts_engine: str = "omnivoice"
    default_asr_engine: str = "whisperx"
    default_diarization_engine: str = "pyannote"
    
    # GPU/Compute
    device: str = "auto"  # auto, cuda, mps, cpu
    fp16: bool = True
    gpu_workers: int = 1
    
    # Timeouts
    generate_timeout_s: int = 300
    cpu_generate_timeout_s: int = 600
    
    # Auth
    remote_api_key: Optional[str] = None
    # Non-loopback Host names accepted (e.g. a LAN IP); these always require remote_api_key.
    allowed_hosts: List[str] = []
    session_ttl_seconds: int = Field(default=3600, gt=0)
    
    # Limits
    max_text_length: int = 5000
    max_audio_duration: int = 300  # seconds
    max_profile_upload_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    clone_max_seconds: int = 15
    ref_hard_max_seconds: int = 75
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # External services
    hf_token: Optional[str] = None
    hf_token_env: bool = False
    
    @property
    def computed_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Constants
IDLE_TIMEOUT_SECONDS = 300
CPU_POOL_WORKERS = 4
