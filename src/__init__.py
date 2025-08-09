# Speech Synthesis Module
"""
Advanced Speech Synthesis package with TTS and voice cloning capabilities.
"""

__version__ = "1.0.0"
__author__ = "Speech Synthesis Team"

# Core modules
from . import model
from . import train  
from . import infer
from . import clone
from . import evaluate
from . import preprocess
from . import export

# Utilities
from . import utils
from . import logging_config
from .logging_config import get_logger, get_simple_logger, log_performance
from . import text_processor
from . import data

# Interface modules
from . import gradio_interface
from . import tts_model
from . import speaker_encoder
from . import vocoder

__all__ = [
    'model',
    'train',
    'infer', 
    'clone',
    'evaluate',
    'preprocess',
    'export',
    'utils',
    'logging_config',
    'get_logger',
    'get_simple_logger',
    'log_performance',
    'text_processor',
    'data',
    'gradio_interface',
    'tts_model',
    'speaker_encoder',
    'vocoder'
]
