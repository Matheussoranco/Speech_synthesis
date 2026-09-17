import numpy as np
import torch

try:
    from TTS.api import TTS
except ImportError:  # pragma: no cover - optional extra [tts]
    TTS = None


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
        if TTS is None:
            raise ImportError(
                "extra tts não instalado: instale com `pip install speech-synthesis[tts]` "
                "para usar o TTSWrapper (Coqui TTS)."
            )
        self.tts = TTS(model_name)

    def synthesize(self, text, speaker_wav=None, speaker=None, language=None):
        """Synthesize *text*, returning a float32 numpy waveform.

        The default model (YourTTS) is both multi-speaker and multi-lingual.
        Coqui raises "Model is multi-speaker but no `speaker` is provided."
        when neither a reference clip nor a speaker name is given, so fall back
        to the model's first built-in speaker/language rather than failing.
        """
        kwargs = {}

        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        else:
            speaker = speaker or self._default_speaker()
            if speaker is not None:
                kwargs["speaker"] = speaker

        language = language or self._default_language()
        if language is not None:
            kwargs["language"] = language

        wav = self.tts.tts(text, **kwargs)
        # Coqui returns a plain list for most models; callers (and save())
        # expect an array.
        return np.asarray(wav, dtype=np.float32)

    def _default_speaker(self):
        speakers = getattr(self.tts, "speakers", None)
        return speakers[0] if speakers else None

    def _default_language(self):
        languages = getattr(self.tts, "languages", None)
        return languages[0] if languages else None

    def save(self, wav, path):
        self.tts.save_wav(wav, path)
