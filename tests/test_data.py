import unittest
from src.data import TTSDataset
import os
import shutil
import numpy as np
import soundfile as sf

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

    def test_dataset(self):
        ds = TTSDataset('tests/tmp', sample_rate=22050)
        audio, text = ds[0]
        self.assertEqual(text, 'hello world')
        self.assertEqual(audio.shape[0], 22050)

if __name__ == '__main__':
    unittest.main()
