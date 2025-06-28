import torch
import torchaudio
from torch.utils.data import Dataset
import os
import librosa

class TTSDataset(Dataset):
    def __init__(self, data_dir, sample_rate=22050):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.items = []
        for fname in os.listdir(data_dir):
            if fname.endswith('.wav'):
                txt = fname.replace('.wav', '.txt')
                if os.path.exists(os.path.join(data_dir, txt)):
                    self.items.append((fname, txt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_file, txt_file = self.items[idx]
        audio, _ = librosa.load(os.path.join(self.data_dir, wav_file), sr=self.sample_rate)
        with open(os.path.join(self.data_dir, txt_file), 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return torch.tensor(audio, dtype=torch.float32), text
