import torch
from TTS.api import TTS

class TTSWrapper:
    """
    Wrapper for Coqui TTS models (Tacotron2, YourTTS, etc.)
    """
    # Coqui model ids are always four segments — type/lang/dataset/model.
    # YourTTS is multilingual, so it lives under multilingual/multi-dataset;
    # the three-segment "tts_models/en/your_tts" makes TTS's ModelManager
    # raise "not enough values to unpack (expected 4, got 3)" on load.
    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/your_tts"

    def __init__(self, model_name=DEFAULT_MODEL):
        self.tts = TTS(model_name)

    def synthesize(self, text, speaker_wav=None):
        if speaker_wav:
            wav = self.tts.tts(text, speaker_wav=speaker_wav)
        else:
            wav = self.tts.tts(text)
        return wav

    def save(self, wav, path):
        self.tts.save_wav(wav, path)
