import torch
from TTS.api import TTS

class TTSWrapper:
    """
    Wrapper for Coqui TTS models (Tacotron2, YourTTS, etc.)
    """
    def __init__(self, model_name="tts_models/en/your_tts"):
        self.tts = TTS(model_name)

    def synthesize(self, text, speaker_wav=None):
        if speaker_wav:
            wav = self.tts.tts(text, speaker_wav=speaker_wav)
        else:
            wav = self.tts.tts(text)
        return wav

    def save(self, wav, path):
        self.tts.save_wav(wav, path)
