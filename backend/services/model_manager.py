"
Model management service - handles loading/unloading models
"

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("omnivoice.model")

_model_cache = {}

async def startup_model_services() -> None:
    """Initialize model services on startup"""
    logger.info("Starting model services...")
    # In production, this would pre-load models
    # For now, just log
    logger.info("Model services started")

async def shutdown_model_services() -> None:
    """Cleanup model services on shutdown"""
    logger.info("Shutting down model services...")
    _model_cache.clear()
    logger.info("Model services stopped")

def get_model(engine_id: str):
    """Get a model instance (with caching)"""
    if engine_id in _model_cache:
        return _model_cache[engine_id]
    
    # In production, load the actual model
    logger.info(f"Loading model: {engine_id}")
    model = {"id": engine_id, "loaded": True}
    _model_cache[engine_id] = model
    return model

def unload_model(engine_id: str) -> bool:
    """Unload a model"""
    if engine_id in _model_cache:
        del _model_cache[engine_id]
        return True
    return False

def list_loaded_models() -> list:
    return list(_model_cache.keys())
