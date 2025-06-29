from resemblyzer import VoiceEncoder
import numpy as np

class SpeakerEncoder:
    def __init__(self, device="cpu"):
        self.encoder = VoiceEncoder().to(device)
        self.device = device

    def extract_embedding(self, wav, sr=16000):
        # wav: numpy array
        return self.encoder.embed_utterance(wav, sample_rate=sr)
