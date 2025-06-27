import torch
import torchaudio
import librosa
import numpy as np

# Utility functions for audio processing, loading, saving, etc.
def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def save_audio(path, audio, sr=22050):
    torchaudio.save(path, torch.tensor(audio).unsqueeze(0), sr)

def preprocess_text(text):
    # Basic text normalization (expand as needed)
    return text.strip().lower()
