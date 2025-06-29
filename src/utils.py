import torch
import torchaudio
import librosa
import numpy as np
from phonemizer import phonemize

# Utility functions for audio processing, loading, saving, etc.
def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def save_audio(path, audio, sr=22050):
    torchaudio.save(path, torch.tensor(audio).unsqueeze(0), sr)

def preprocess_text(text, lang='en'):
    # Normalize and phonemize text
    text = text.strip().lower()
    phonemes = phonemize(text, language=lang, backend='espeak')
    return phonemes
