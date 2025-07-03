"""
Utility functions and classes for the Speech Synthesis system.
Includes helper functions, training utilities, and common operations.
"""
import torch
import torch.nn as nn
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import yaml
import os
from pathlib import Path
import time
import hashlib
from functools import wraps
import logging
from phonemizer import phonemize


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class GradientClipper:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Args:
            max_norm: Maximum norm of gradients
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, parameters):
        """Clip gradients of parameters."""
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)


class LearningRateScheduler:
    """Custom learning rate scheduler with warmup and decay."""
    
    def __init__(self, optimizer, warmup_steps: int = 4000, d_model: int = 512):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            d_model: Model dimension for scaling
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0
        
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self) -> float:
        """Calculate learning rate."""
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )


# Legacy functions for backward compatibility
def load_audio(path, sr=22050):
    """Load audio file - legacy function."""
    audio, _ = librosa.load(path, sr=sr)
    return audio


def save_audio(path, audio, sr=22050):
    """Save audio file - legacy function."""
    torchaudio.save(path, torch.tensor(audio).unsqueeze(0), sr)


def preprocess_text(text, lang='en'):
    """Preprocess text - legacy function."""
    # Normalize and phonemize text
    text = text.strip().lower()
    phonemes = phonemize(text, language=lang, backend='espeak')
    return phonemes


class AudioProcessor:
    """Audio processing utilities."""
    
    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 22050) -> np.ndarray:
        """Load audio file."""
        audio, sr = librosa.load(file_path, sr=sample_rate)
        return audio
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 22050):
        """Save audio to file."""
        sf.write(file_path, audio, sample_rate)
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Normalize audio to target RMS level."""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            return audio * (target_rms / current_rms)
        return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) > 0:
            return audio[non_silent[0]:non_silent[-1] + 1]
        return audio
    
    @staticmethod
    def compute_mel_spectrogram(audio: np.ndarray, 
                               sample_rate: int = 22050,
                               n_fft: int = 1024,
                               hop_length: int = 256,
                               n_mels: int = 80) -> np.ndarray:
        """Compute mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)


class ModelUtils:
    """Model utilities."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, Any]:
        """Get detailed model size information."""
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Estimate model size in MB
        model_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb
        }


def setup_reproducibility(seed: int = 42):
    """Setup reproducible random number generation."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)
