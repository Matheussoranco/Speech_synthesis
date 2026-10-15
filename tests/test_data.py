import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data import TTSDataset
from src.text_processor import TextProcessor
from conftest import requires_espeak


class TestTTSDataset(unittest.TestCase):
    @requires_espeak
    def test_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = np.random.default_rng(42).normal(0, 0.1, 22050)
            sf.write(root / '0.wav', audio, 22050)
            (root / '0.txt').write_text('hello world', encoding='utf-8')
            processor = TextProcessor(language='en')
            expected = processor.phonemes_to_ids(processor.text_to_phonemes('hello world'))
            item = TTSDataset(directory, sample_rate=22050, text_processor=processor)[0]
            self.assertEqual(item['text_tokens'].tolist(), expected)
            self.assertEqual(item['wav'].shape[0], 22050)

    def test_stereo_wav_is_averaged_and_resampled_without_ffmpeg(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = np.full(16000, 0.2, dtype=np.float32)
            sf.write(root / 'stereo.wav', np.column_stack((left, -left)), 16000, subtype='FLOAT')
            (root / 'stereo.txt').write_text('audio', encoding='utf-8')
            processor = Mock()
            processor.text_to_phonemes.return_value = 'a'
            processor.phonemes_to_ids.return_value = [1]
            item = TTSDataset(directory, sample_rate=22050, text_processor=processor)[0]
            self.assertEqual(item['wav'].shape[0], 22050)
            self.assertEqual(item['wav'].abs().max().item(), 0)
            self.assertTrue(np.isfinite(item['spectrogram'].numpy()).all())
            (root / 'stereo.wav').write_bytes(b'not audio')
            with self.assertRaises(sf.LibsndfileError):
                TTSDataset(directory, text_processor=processor)[0]


if __name__ == '__main__':
    unittest.main()
