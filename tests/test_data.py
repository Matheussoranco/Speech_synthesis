import unittest
from src.data import TTSDataset
import os
import shutil
import numpy as np
import soundfile as sf

from conftest import requires_ffmpeg

class TestTTSDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs('tests/tmp', exist_ok=True)
        # Create dummy wav and txt
        audio = np.random.randn(22050)
        sf.write('tests/tmp/0.wav', audio, 22050)
        with open('tests/tmp/0.txt', 'w') as f:
            f.write('hello world')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tests/tmp')

    @requires_ffmpeg
    def test_dataset(self):
        # __getitem__ returns a dict (text_tokens/spectrogram/wav/...), not
        # the (audio, text) tuple this test used to unpack -- that API
        # predates collate_fn's dict-based batching and no longer exists.
        ds = TTSDataset('tests/tmp', sample_rate=22050)
        item = ds[0]
        self.assertEqual(item['text_tokens'].shape[0], len('hello world'))
        self.assertEqual(item['wav'].shape[0], 22050)

if __name__ == '__main__':
    unittest.main()
