import unittest
from src.tts_model import TTSWrapper
from src.speaker_encoder import SpeakerEncoder
import numpy as np

class TestTTSIntegration(unittest.TestCase):
    def test_tts_synthesize(self):
        tts = TTSWrapper()
        wav = tts.synthesize("test")
        self.assertIsInstance(wav, np.ndarray)

    def test_speaker_encoder(self):
        encoder = SpeakerEncoder()
        dummy_wav = np.random.randn(16000)
        emb = encoder.extract_embedding(dummy_wav, sr=16000)
        self.assertEqual(emb.shape[-1], 256)

if __name__ == '__main__':
    unittest.main()
