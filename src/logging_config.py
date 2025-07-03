"""
Advanced logging configuration for Speech Synthesis system.
Provides structured logging with different levels and output formats.
"""
import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
import json
from datetime import datetime


class SpeechSynthesisLogger:
    """Enhanced logger for speech synthesis operations."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "./logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        Initialize logger with custom configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
        """
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Configure console logging
        if enable_console_logging:
            logger.add(
                sys.stdout,
                level=log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
        
        # Configure file logging
        if enable_file_logging:
            # General log file
            logger.add(
                self.log_dir / "speech_synthesis.log",
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="10 MB",
                retention="30 days",
                compression="zip"
            )
            
            # Error log file
            logger.add(
                self.log_dir / "errors.log",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="5 MB",
                retention="90 days"
            )
            
            # Training log file
            logger.add(
                self.log_dir / "training.log",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                filter=lambda record: "training" in record["extra"],
                rotation="50 MB",
                retention="180 days"
            )
    
    def log_training_metrics(self, epoch: int, metrics: dict):
        """Log training metrics with structured format."""
        logger.bind(training=True).info(
            f"Epoch {epoch} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
    
    def log_inference_time(self, text_length: int, inference_time: float, model_type: str):
        """Log inference performance metrics."""
        logger.info(
            f"Inference | Model: {model_type} | Text length: {text_length} chars | "
            f"Time: {inference_time:.3f}s | Speed: {text_length/inference_time:.1f} chars/s"
        )
    
    def log_model_info(self, model_name: str, num_parameters: int, device: str):
        """Log model information."""
        logger.info(
            f"Model loaded | Name: {model_name} | Parameters: {num_parameters:,} | Device: {device}"
        )
    
    def log_audio_processing(self, input_file: str, output_file: str, duration: float):
        """Log audio processing operations."""
        logger.info(
            f"Audio processed | Input: {input_file} | Output: {output_file} | Duration: {duration:.2f}s"
        )
    
    def log_error_with_context(self, error: Exception, context: dict):
        """Log error with additional context."""
        logger.error(
            f"Error: {str(error)} | Context: {json.dumps(context, default=str)}"
        )
    
    def log_voice_cloning(self, reference_audio: str, target_text: str, similarity_score: float):
        """Log voice cloning operations."""
        logger.info(
            f"Voice cloning | Reference: {reference_audio} | "
            f"Text length: {len(target_text)} | Similarity: {similarity_score:.3f}"
        )


# Global logger instance
def get_logger(config: Optional[dict] = None) -> SpeechSynthesisLogger:
    """Get configured logger instance."""
    if config is None:
        config = {}
    
    return SpeechSynthesisLogger(
        log_level=config.get("log_level", "INFO"),
        log_dir=config.get("log_dir", "./logs"),
        enable_file_logging=config.get("enable_file_logging", True),
        enable_console_logging=config.get("enable_console_logging", True)
    )


# Performance logging decorator
def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    return wrapper
