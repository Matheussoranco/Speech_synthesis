"""
Advanced dataset classes and data loading utilities for TTS training.
Supports various data formats, augmentation, and preprocessing.
"""
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import soundfile as sf
import pandas as pd
from omegaconf import DictConfig


class TTSDataset(Dataset):
    """
    Advanced TTS dataset with comprehensive preprocessing and augmentation.
    Supports multiple data formats and advanced audio processing.
    """
    
    def __init__(self, 
                 data_dir: str,
                 sample_rate: int = 22050,
                 text_processor: Optional[Any] = None,
                 config: Optional[DictConfig] = None,
                 subset: str = "train"):
        """
        Initialize TTS dataset.
        
        Args:
            data_dir: Directory containing audio and text files
            sample_rate: Target sample rate
            text_processor: Text processing instance
            config: Data configuration
            subset: Dataset subset (train/val/test)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.text_processor = text_processor
        self.config = config or {}
        self.subset = subset
        
        # Audio processing parameters
        self.n_fft = getattr(config, 'win_length', 1024)
        self.hop_length = getattr(config, 'hop_length', 256)
        self.n_mel_channels = getattr(config, 'n_mel_channels', 80)
        self.mel_fmin = getattr(config, 'mel_fmin', 0)
        self.mel_fmax = getattr(config, 'mel_fmax', 8000)
        
        # Data augmentation settings
        self.use_augmentation = subset == "train" and getattr(config, 'use_augmentation', False)
        self.normalize_audio = getattr(config, 'normalize_audio', True)
        self.trim_silence = getattr(config, 'trim_silence', True)
        
        # Load data items
        self.items = self._load_data_items()
        
        # Precompute mel spectrograms if configured
        self.precompute_mels = getattr(config, 'precompute_mels', False)
        if self.precompute_mels:
            self._precompute_mel_spectrograms()
    
    def _load_data_items(self) -> List[Dict[str, Any]]:
        """Load data items from directory."""
        items = []
        
        # Check for metadata file
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            items = self._load_from_metadata(metadata_file)
        else:
            items = self._load_from_directory()
        
        # Filter items based on criteria
        items = self._filter_items(items)
        
        print(f"Loaded {len(items)} items for {self.subset} set from {self.data_dir}")
        return items
    
    def _load_from_metadata(self, metadata_file: Path) -> List[Dict[str, Any]]:
        """Load items from metadata JSON file."""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        items = []
        for item in metadata:
            audio_path = self.data_dir / item['audio_file']
            if audio_path.exists():
                items.append({
                    'audio_path': str(audio_path),
                    'text': item['text'],
                    'speaker_id': item.get('speaker_id', 0),
                    'duration': item.get('duration', None),
                    'language': item.get('language', 'en')
                })
        
        return items
    
    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """Load items by scanning directory for audio/text pairs."""
        items = []
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        
        for audio_file in self.data_dir.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                # Look for corresponding text file
                text_file = audio_file.with_suffix('.txt')
                
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    # Extract speaker ID from path if available
                    speaker_id = self._extract_speaker_id(audio_file)
                    
                    items.append({
                        'audio_path': str(audio_file),
                        'text': text,
                        'speaker_id': speaker_id,
                        'duration': None,
                        'language': 'en'
                    })
        
        return items
    
    def _extract_speaker_id(self, audio_path: Path) -> int:
        """Extract speaker ID from file path."""
        # Look for speaker information in parent directory names
        for part in audio_path.parts:
            if part.startswith('speaker_') or part.startswith('spk_'):
                try:
                    return int(part.split('_')[1])
                except (IndexError, ValueError):
                    pass
        return 0
    
    def _filter_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter items based on duration and other criteria."""
        filtered_items = []
        
        min_duration = getattr(self.config, 'min_duration', 0.5)
        max_duration = getattr(self.config, 'max_duration', 20.0)
        max_text_length = getattr(self.config, 'max_text_length', 500)
        
        for item in items:
            # Check text length
            if len(item['text']) > max_text_length:
                continue
            
            # Check audio duration if available
            if item['duration'] is not None:
                if item['duration'] < min_duration or item['duration'] > max_duration:
                    continue
            else:
                # Compute duration
                try:
                    audio_info = torchaudio.info(item['audio_path'])
                    duration = audio_info.num_frames / audio_info.sample_rate
                    item['duration'] = duration
                    
                    if duration < min_duration or duration > max_duration:
                        continue
                except Exception:
                    continue
            
            filtered_items.append(item)
        
        return filtered_items
    
    def _precompute_mel_spectrograms(self):
        """Precompute mel spectrograms for faster training."""
        print("Precomputing mel spectrograms...")
        
        cache_dir = self.data_dir / "mel_cache"
        cache_dir.mkdir(exist_ok=True)
        
        for i, item in enumerate(self.items):
            cache_file = cache_dir / f"mel_{i}.npy"
            
            if not cache_file.exists():
                audio = self._load_audio(item['audio_path'])
                mel_spec = self._compute_mel_spectrogram(audio)
                np.save(cache_file, mel_spec)
            
            item['mel_cache_path'] = str(cache_file)
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Trim silence
        if self.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        if self.normalize_audio:
            audio = librosa.util.normalize(audio)
        
        return audio
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec.T  # (time, mel_channels)
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentation."""
        if not self.use_augmentation:
            return audio
        
        # Random volume scaling
        if random.random() < 0.3:
            volume_factor = random.uniform(0.7, 1.3)
            audio = audio * volume_factor
        
        # Add noise
        if random.random() < 0.2:
            noise_factor = random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_factor, audio.shape)
            audio = audio + noise
        
        # Time stretching
        if random.random() < 0.1:
            stretch_factor = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        # Ensure audio doesn't clip
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _process_text(self, text: str) -> torch.Tensor:
        """Process text into tokens."""
        if self.text_processor:
            processed_text = self.text_processor.process_text(
                text,
                use_phonemes=getattr(self.config, 'use_phonemes', False)
            )
            tokens = self.text_processor.tokenize_text(processed_text)
        else:
            # Simple character-level tokenization
            tokens = list(text.lower())
        
        # Convert to indices (simplified)
        char_to_idx = {chr(i): i for i in range(32, 127)}  # Printable ASCII
        token_ids = [char_to_idx.get(char, 0) for char in tokens]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        item = self.items[idx]
        
        # Load audio
        if self.precompute_mels and 'mel_cache_path' in item:
            mel_spectrogram = np.load(item['mel_cache_path'])
        else:
            audio = self._load_audio(item['audio_path'])
            audio = self._augment_audio(audio)
            mel_spectrogram = self._compute_mel_spectrogram(audio)
        
        # Process text
        text_tokens = self._process_text(item['text'])
        
        # Create output dictionary
        output = {
            'audio_path': item['audio_path'],
            'text': item['text'],
            'text_tokens': text_tokens,
            'mel_spectrogram': torch.tensor(mel_spectrogram, dtype=torch.float32),
            'speaker_id': torch.tensor(item['speaker_id'], dtype=torch.long),
            'duration': torch.tensor(item['duration'], dtype=torch.float32)
        }
        
        return output


class TTSCollator:
    """Collate function for TTS data loader."""
    
    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator.
        
        Args:
            pad_token_id: Token ID used for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Extract sequences
        text_tokens = [item['text_tokens'] for item in batch]
        mel_spectrograms = [item['mel_spectrogram'] for item in batch]
        
        # Pad text sequences
        text_lengths = torch.tensor([len(seq) for seq in text_tokens])
        max_text_len = text_lengths.max().item()
        
        padded_text = torch.full(
            (len(batch), max_text_len),
            self.pad_token_id,
            dtype=torch.long
        )
        
        for i, seq in enumerate(text_tokens):
            padded_text[i, :len(seq)] = seq
        
        # Pad mel spectrograms
        mel_lengths = torch.tensor([mel.size(0) for mel in mel_spectrograms])
        max_mel_len = mel_lengths.max().item()
        mel_channels = mel_spectrograms[0].size(1)
        
        padded_mels = torch.zeros(
            len(batch), max_mel_len, mel_channels,
            dtype=torch.float32
        )
        
        for i, mel in enumerate(mel_spectrograms):
            padded_mels[i, :mel.size(0)] = mel
        
        # Create attention masks
        text_mask = torch.arange(max_text_len)[None, :] < text_lengths[:, None]
        mel_mask = torch.arange(max_mel_len)[None, :] < mel_lengths[:, None]
        
        # Collect other data
        speaker_ids = torch.stack([item['speaker_id'] for item in batch])
        durations = torch.stack([item['duration'] for item in batch])
        
        return {
            'text_tokens': padded_text,
            'text_lengths': text_lengths,
            'text_mask': text_mask,
            'mel_spectrogram': padded_mels,
            'mel_lengths': mel_lengths,
            'mel_mask': mel_mask,
            'speaker_id': speaker_ids,
            'duration': durations,
            'texts': [item['text'] for item in batch],
            'audio_paths': [item['audio_path'] for item in batch]
        }


def create_dataloader(dataset: TTSDataset,
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = True) -> DataLoader:
    """Create data loader with appropriate collator."""
    collator = TTSCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator
    )


class MultiSpeakerTTSDataset(TTSDataset):
    """Dataset for multi-speaker TTS training."""
    
    def __init__(self, *args, **kwargs):
        """Initialize multi-speaker dataset."""
        super().__init__(*args, **kwargs)
        
        # Build speaker mapping
        self.speaker_mapping = self._build_speaker_mapping()
        self.num_speakers = len(self.speaker_mapping)
        
        print(f"Found {self.num_speakers} unique speakers")
    
    def _build_speaker_mapping(self) -> Dict[int, int]:
        """Build mapping from original speaker IDs to contiguous IDs."""
        unique_speakers = set(item['speaker_id'] for item in self.items)
        return {spk_id: idx for idx, spk_id in enumerate(sorted(unique_speakers))}
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with mapped speaker ID."""
        output = super().__getitem__(idx)
        
        # Map speaker ID to contiguous range
        original_speaker_id = output['speaker_id'].item()
        mapped_speaker_id = self.speaker_mapping[original_speaker_id]
        output['speaker_id'] = torch.tensor(mapped_speaker_id, dtype=torch.long)
        
        return output


class VoiceCloningDataset(Dataset):
    """Dataset for voice cloning training."""
    
    def __init__(self,
                 data_dir: str,
                 reference_length: float = 3.0,
                 target_length: float = 5.0,
                 sample_rate: int = 22050,
                 config: Optional[DictConfig] = None):
        """
        Initialize voice cloning dataset.
        
        Args:
            data_dir: Directory containing speaker data
            reference_length: Length of reference audio in seconds
            target_length: Length of target audio in seconds
            sample_rate: Target sample rate
            config: Data configuration
        """
        self.data_dir = Path(data_dir)
        self.reference_length = reference_length
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.config = config or {}
        
        # Load speaker data
        self.speaker_data = self._load_speaker_data()
    
    def _load_speaker_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load data organized by speaker."""
        speaker_data = {}
        
        for speaker_dir in self.data_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                speaker_data[speaker_id] = []
                
                for audio_file in speaker_dir.glob('*.wav'):
                    text_file = audio_file.with_suffix('.txt')
                    
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        speaker_data[speaker_id].append({
                            'audio_path': str(audio_file),
                            'text': text
                        })
        
        return speaker_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return sum(len(items) for items in self.speaker_data.values())
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item for voice cloning."""
        # Select speaker and utterances
        speaker_id = list(self.speaker_data.keys())[idx % len(self.speaker_data)]
        utterances = self.speaker_data[speaker_id]
        
        # Randomly select reference and target utterances
        if len(utterances) >= 2:
            ref_idx, target_idx = random.sample(range(len(utterances)), 2)
        else:
            ref_idx = target_idx = 0
        
        ref_item = utterances[ref_idx]
        target_item = utterances[target_idx]
        
        # Load and process audio
        ref_audio = self._load_audio_segment(
            ref_item['audio_path'], 
            self.reference_length
        )
        target_audio = self._load_audio_segment(
            target_item['audio_path'],
            self.target_length
        )
        
        return {
            'reference_audio': torch.tensor(ref_audio, dtype=torch.float32),
            'target_audio': torch.tensor(target_audio, dtype=torch.float32),
            'target_text': target_item['text'],
            'speaker_id': speaker_id
        }
    
    def _load_audio_segment(self, audio_path: str, duration: float) -> np.ndarray:
        """Load audio segment of specified duration."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        target_samples = int(duration * self.sample_rate)
        
        if len(audio) >= target_samples:
            # Random crop
            start_idx = random.randint(0, len(audio) - target_samples)
            audio = audio[start_idx:start_idx + target_samples]
        else:
            # Pad with silence
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
