"""
TTS Dataset — supports LJSpeech, VCTK, LibriTTS, Common Voice, generic.

Each item returned by __getitem__ contains:
  text_tokens   : LongTensor (T_text,)
  text_lengths  : int
  spectrogram   : FloatTensor (n_fft//2+1, T_spec)  — linear spectrogram
  spec_lengths  : int
  wav           : FloatTensor (T_wav,)
  speaker_id    : int | None
  speaker_embedding: FloatTensor (emb_dim,) | None
"""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = dict


class TTSDataset(Dataset):
    """
    Generic TTS dataset.

    Expects one of:
      - LJSpeech   metadata.csv  with  |filename|normalised|text|
      - VCTK       speaker dirs  with  *.wav + *.txt
      - LibriTTS   *.trans.tsv   or  auto-detected paired .wav/.txt
      - Generic    paired audio/text files, named identically
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        text_processor=None,
        config: Optional[Any] = None,
        subset: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.tp = text_processor
        self.config = config or {}
        self.subset = subset

        self.n_fft = int(getattr(config, "n_fft", 1024))
        self.hop_length = int(getattr(config, "hop_length", 256))
        self.win_length = int(getattr(config, "win_length", 1024))
        self.n_mels = int(getattr(config, "n_mel_channels", 80))
        self.f_min = float(getattr(config, "mel_fmin", 0.0))
        self.f_max = float(getattr(config, "mel_fmax", 8000.0))
        self.max_wav_len = int(getattr(config, "max_wav_length", 220500))  # 10 s

        self.samples: List[Dict] = []
        self.speaker2id: Dict[str, int] = {}
        self._load_metadata()

        # Window for STFT
        self.window = torch.hann_window(self.win_length)

    # ------------------------------------------------------------------
    def _load_metadata(self):
        if not self.data_dir.exists():
            return

        # 1. LJSpeech
        meta_csv = self.data_dir / "metadata.csv"
        if meta_csv.exists():
            self._load_ljspeech(meta_csv)
            return

        # 2. LibriTTS / generic .trans.tsv
        for tsv in self.data_dir.rglob("*.trans.tsv"):
            self._load_trans_tsv(tsv)
        if self.samples:
            return

        # 3. VCTK — speaker subdirectories
        wav_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if wav_dirs:
            for spk_dir in sorted(wav_dirs):
                spk_id = spk_dir.name
                if spk_id not in self.speaker2id:
                    self.speaker2id[spk_id] = len(self.speaker2id)
                for wav_path in sorted(spk_dir.glob("*.wav")):
                    txt_path = wav_path.with_suffix(".txt")
                    if txt_path.exists():
                        self.samples.append({
                            "wav": str(wav_path),
                            "text": txt_path.read_text(encoding="utf-8").strip(),
                            "speaker": spk_id,
                        })
            if self.samples:
                return

        # 4. Flat directory of paired .wav + .txt
        for wav_path in sorted(self.data_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                self.samples.append({
                    "wav": str(wav_path),
                    "text": txt_path.read_text(encoding="utf-8").strip(),
                    "speaker": "default",
                })
        if "default" not in self.speaker2id:
            self.speaker2id["default"] = 0

    def _load_ljspeech(self, csv_path: Path):
        """LJSpeech: filename|raw|normalised"""
        wavs_dir = csv_path.parent / "wavs"
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                name = parts[0]
                text = parts[2] if len(parts) >= 3 else parts[1]
                wav_p = wavs_dir / f"{name}.wav"
                if wav_p.exists():
                    self.samples.append({
                        "wav": str(wav_p),
                        "text": text,
                        "speaker": "LJ",
                    })
        self.speaker2id = {"LJ": 0}

    def _load_trans_tsv(self, tsv: Path):
        """LibriTTS *.trans.tsv: wav_id<TAB>transcript"""
        parent = tsv.parent
        spk = parent.parent.parent.name  # books/chapter/spk structure
        if spk not in self.speaker2id:
            self.speaker2id[spk] = len(self.speaker2id)
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                wav_id, text = parts[0], parts[1]
                wav_p = parent / f"{wav_id}.wav"
                if wav_p.exists():
                    self.samples.append({
                        "wav": str(wav_p), "text": text, "speaker": spk
                    })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Text → phoneme IDs
        text = sample["text"]
        if self.tp is not None:
            try:
                phonemes = self.tp.text_to_phonemes(text)
                ids = self.tp.phonemes_to_ids(phonemes)
            except Exception:
                ids = [ord(c) % 256 for c in text]
        else:
            ids = [ord(c) % 256 for c in text]

        text_tokens = torch.LongTensor(ids)

        # Load audio
        wav, sr = torchaudio.load(sample["wav"])
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Trim / pad
        if len(wav) > self.max_wav_len:
            start = random.randint(0, len(wav) - self.max_wav_len)
            wav = wav[start : start + self.max_wav_len]

        # Linear spectrogram (for posterior encoder)
        spec = self._compute_spec(wav)

        # Speaker
        spk_name = sample.get("speaker", "default")
        spk_id = self.speaker2id.get(spk_name, 0)

        return {
            "text_tokens": text_tokens,
            "text_lengths": len(ids),
            "spectrogram": spec,
            "spec_lengths": spec.shape[-1],
            "wav": wav,
            "speaker_id": spk_id,
        }

    def _compute_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute linear magnitude spectrogram (n_fft//2+1, T)."""
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        spec = torch.abs(stft)
        return spec  # (n_fft//2+1, T)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Pad all variable-length tensors in a batch.

    text_tokens  → (B, max_T_text)
    spectrogram  → (B, n_fft//2+1, max_T_spec)
    wav          → (B, max_T_wav)
    """
    text_tokens = [b["text_tokens"] for b in batch]
    spectrograms = [b["spectrogram"] for b in batch]
    wavs = [b["wav"] for b in batch]
    text_lengths = torch.LongTensor([b["text_lengths"] for b in batch])
    spec_lengths = torch.LongTensor([b["spec_lengths"] for b in batch])
    speaker_ids = torch.LongTensor([b["speaker_id"] for b in batch])

    # Pad text
    max_tl = max(len(t) for t in text_tokens)
    text_pad = torch.zeros(len(batch), max_tl, dtype=torch.long)
    for i, t in enumerate(text_tokens):
        text_pad[i, : len(t)] = t

    # Pad spec
    freq = spectrograms[0].shape[0]
    max_sl = max(s.shape[1] for s in spectrograms)
    spec_pad = torch.zeros(len(batch), freq, max_sl)
    for i, s in enumerate(spectrograms):
        spec_pad[i, :, : s.shape[1]] = s

    # Pad wav
    max_wl = max(w.shape[0] for w in wavs)
    wav_pad = torch.zeros(len(batch), max_wl)
    for i, w in enumerate(wavs):
        wav_pad[i, : w.shape[0]] = w

    return {
        "text_tokens": text_pad,
        "text_lengths": text_lengths,
        "spectrogram": spec_pad,
        "spec_lengths": spec_lengths,
        "wav": wav_pad,
        "speaker_id": speaker_ids,
    }


def create_dataloader(dataset: Dataset, config, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.system.get("max_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )
