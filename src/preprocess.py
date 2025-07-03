#!/usr/bin/env python3
"""
Data Preprocessing Module

Provides comprehensive data preprocessing functionality for TTS datasets including:
- Audio preprocessing (normalization, resampling, etc.)
- Text preprocessing (cleaning, normalization, phonemization)
- Dataset creation and validation
- Multi-speaker dataset handling
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    import torch
    import torchaudio
    import numpy as np
    import librosa
    from tqdm import tqdm
    from omegaconf import DictConfig, OmegaConf
    TORCH_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Some dependencies not available: {e}")
    TORCH_AVAILABLE = False

from .logging_config import get_logger
from .text_processor import TextProcessor
from .utils import AudioProcessor

logger = get_logger()


class DatasetPreprocessor:
    """Comprehensive dataset preprocessing system."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.text_processor = TextProcessor(config)
        self.audio_processor = AudioProcessor()
        
        # Audio settings
        self.target_sample_rate = config.audio.sample_rate
        self.max_audio_length = getattr(config.audio, 'max_length', 10.0)  # seconds
        self.min_audio_length = getattr(config.audio, 'min_length', 0.5)   # seconds
        
    def preprocess_ljspeech(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Preprocess LJSpeech dataset."""
        logger.info("Preprocessing LJSpeech dataset")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Paths
        metadata_file = input_path / "metadata.csv"
        wavs_dir = input_path / "wavs"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        if not wavs_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {wavs_dir}")
        
        # Create output directories
        (output_path / "audio").mkdir(exist_ok=True)
        (output_path / "spectrograms").mkdir(exist_ok=True)
        
        # Read metadata
        metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    audio_id = parts[0]
                    text = parts[2] if len(parts) > 2 else parts[1]
                    metadata.append((audio_id, text))
        
        # Process samples
        processed_samples = []
        stats = {
            'total_samples': len(metadata),
            'successful_samples': 0,
            'failed_samples': 0,
            'audio_stats': {
                'durations': [],
                'sample_rates': [],
                'channels': []
            }
        }
        
        for audio_id, text in tqdm(metadata, desc="Processing LJSpeech"):
            try:
                # Process audio
                audio_path = wavs_dir / f"{audio_id}.wav"
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    stats['failed_samples'] += 1
                    continue
                
                processed_audio, audio_info = self._process_audio_file(
                    audio_path, output_path / "audio" / f"{audio_id}.wav"
                )
                
                if processed_audio is None:
                    stats['failed_samples'] += 1
                    continue
                
                # Process text
                processed_text = self.text_processor.process(text)
                
                # Create spectrogram
                spectrogram_path = output_path / "spectrograms" / f"{audio_id}.pt"
                self._create_spectrogram(processed_audio, spectrogram_path)
                
                # Store sample info
                sample_info = {
                    'id': audio_id,
                    'text': text,
                    'processed_text': processed_text,
                    'audio_path': str(output_path / "audio" / f"{audio_id}.wav"),
                    'spectrogram_path': str(spectrogram_path),
                    'duration': audio_info['duration'],
                    'sample_rate': audio_info['sample_rate']
                }
                
                processed_samples.append(sample_info)
                stats['successful_samples'] += 1
                stats['audio_stats']['durations'].append(audio_info['duration'])
                stats['audio_stats']['sample_rates'].append(audio_info['sample_rate'])
                stats['audio_stats']['channels'].append(audio_info['channels'])
                
            except Exception as e:
                logger.error(f"Failed to process sample {audio_id}: {e}")
                stats['failed_samples'] += 1
        
        # Save processed dataset info
        dataset_info = {
            'dataset_type': 'ljspeech',
            'samples': processed_samples,
            'statistics': stats,
            'config': OmegaConf.to_yaml(self.config)
        }
        
        with open(output_path / "dataset.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create train/val splits
        self._create_splits(processed_samples, output_path)
        
        logger.info(f"LJSpeech preprocessing completed: {stats['successful_samples']}/{stats['total_samples']} samples")
        return dataset_info
    
    def preprocess_custom_dataset(self, input_dir: str, output_dir: str, 
                                metadata_format: str = "auto") -> Dict[str, Any]:
        """Preprocess custom dataset."""
        logger.info("Preprocessing custom dataset")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Detect dataset format
        if metadata_format == "auto":
            metadata_format = self._detect_dataset_format(input_path)
        
        if metadata_format == "ljspeech":
            return self.preprocess_ljspeech(input_dir, output_dir)
        elif metadata_format == "common_voice":
            return self._preprocess_common_voice(input_path, output_path)
        elif metadata_format == "vctk":
            return self._preprocess_vctk(input_path, output_path)
        else:
            return self._preprocess_generic(input_path, output_path)
    
    def _detect_dataset_format(self, input_path: Path) -> str:
        """Auto-detect dataset format."""
        if (input_path / "metadata.csv").exists():
            return "ljspeech"
        elif (input_path / "validated.tsv").exists():
            return "common_voice"
        elif any(input_path.glob("p*/p*_*.wav")):
            return "vctk"
        else:
            return "generic"
    
    def _preprocess_common_voice(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Preprocess Common Voice dataset."""
        logger.info("Processing Common Voice format")
        
        # Read TSV file
        tsv_file = input_path / "validated.tsv"
        if not tsv_file.exists():
            tsv_file = input_path / "train.tsv"
        
        if not tsv_file.exists():
            raise FileNotFoundError("No TSV metadata file found")
        
        import csv
        samples = []
        
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                audio_path = input_path / "clips" / row['path']
                if audio_path.exists():
                    samples.append({
                        'audio_path': audio_path,
                        'text': row['sentence'],
                        'id': row['path'].replace('.mp3', '').replace('.wav', '')
                    })
        
        return self._process_samples(samples, output_path, "common_voice")
    
    def _preprocess_vctk(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Preprocess VCTK dataset."""
        logger.info("Processing VCTK format")
        
        # Find all audio files
        audio_files = list(input_path.glob("wav48_silence_trimmed/*/*/*.flac"))
        if not audio_files:
            audio_files = list(input_path.glob("wav48/*/*/*.wav"))
        
        samples = []
        txt_dir = input_path / "txt"
        
        for audio_path in audio_files:
            # Extract speaker and utterance ID
            speaker_id = audio_path.parent.name
            utterance_id = audio_path.stem
            
            # Find corresponding text file
            txt_file = txt_dir / speaker_id / f"{utterance_id}.txt"
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                samples.append({
                    'audio_path': audio_path,
                    'text': text,
                    'id': f"{speaker_id}_{utterance_id}",
                    'speaker_id': speaker_id
                })
        
        return self._process_samples(samples, output_path, "vctk")
    
    def _preprocess_generic(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Preprocess generic dataset (audio files + text files or JSON)."""
        logger.info("Processing generic format")
        
        samples = []
        
        # Look for JSON metadata
        json_files = list(input_path.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if isinstance(item, dict) and 'audio' in item and 'text' in item:
                    audio_path = input_path / item['audio']
                    if audio_path.exists():
                        samples.append({
                            'audio_path': audio_path,
                            'text': item['text'],
                            'id': item.get('id', audio_path.stem),
                            'speaker_id': item.get('speaker_id')
                        })
        else:
            # Look for paired audio and text files
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(input_path.glob(f"*{ext}"))
            
            for audio_path in audio_files:
                txt_path = audio_path.with_suffix('.txt')
                if txt_path.exists():
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    samples.append({
                        'audio_path': audio_path,
                        'text': text,
                        'id': audio_path.stem
                    })
        
        return self._process_samples(samples, output_path, "generic")
    
    def _process_samples(self, samples: List[Dict], output_path: Path, 
                        dataset_type: str) -> Dict[str, Any]:
        """Process a list of samples."""
        # Create output directories
        (output_path / "audio").mkdir(exist_ok=True)
        (output_path / "spectrograms").mkdir(exist_ok=True)
        
        processed_samples = []
        stats = {
            'total_samples': len(samples),
            'successful_samples': 0,
            'failed_samples': 0,
            'audio_stats': {
                'durations': [],
                'sample_rates': [],
                'channels': []
            }
        }
        
        for sample in tqdm(samples, desc=f"Processing {dataset_type}"):
            try:
                # Process audio
                output_audio_path = output_path / "audio" / f"{sample['id']}.wav"
                processed_audio, audio_info = self._process_audio_file(
                    sample['audio_path'], output_audio_path
                )
                
                if processed_audio is None:
                    stats['failed_samples'] += 1
                    continue
                
                # Process text
                processed_text = self.text_processor.process(sample['text'])
                
                # Create spectrogram
                spectrogram_path = output_path / "spectrograms" / f"{sample['id']}.pt"
                self._create_spectrogram(processed_audio, spectrogram_path)
                
                # Store sample info
                sample_info = {
                    'id': sample['id'],
                    'text': sample['text'],
                    'processed_text': processed_text,
                    'audio_path': str(output_audio_path),
                    'spectrogram_path': str(spectrogram_path),
                    'duration': audio_info['duration'],
                    'sample_rate': audio_info['sample_rate']
                }
                
                if 'speaker_id' in sample:
                    sample_info['speaker_id'] = sample['speaker_id']
                
                processed_samples.append(sample_info)
                stats['successful_samples'] += 1
                stats['audio_stats']['durations'].append(audio_info['duration'])
                stats['audio_stats']['sample_rates'].append(audio_info['sample_rate'])
                stats['audio_stats']['channels'].append(audio_info['channels'])
                
            except Exception as e:
                logger.error(f"Failed to process sample {sample['id']}: {e}")
                stats['failed_samples'] += 1
        
        # Save processed dataset info
        dataset_info = {
            'dataset_type': dataset_type,
            'samples': processed_samples,
            'statistics': stats,
            'config': OmegaConf.to_yaml(self.config)
        }
        
        with open(output_path / "dataset.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create train/val splits
        self._create_splits(processed_samples, output_path)
        
        logger.info(f"{dataset_type} preprocessing completed: {stats['successful_samples']}/{stats['total_samples']} samples")
        return dataset_info
    
    def _process_audio_file(self, input_path: Path, output_path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Process a single audio file."""
        try:
            # Load audio
            if TORCH_AVAILABLE:
                waveform, sample_rate = torchaudio.load(str(input_path))
                audio = waveform.squeeze().numpy()
            else:
                audio, sample_rate = librosa.load(str(input_path), sr=None)
            
            # Get original info
            duration = len(audio) / sample_rate
            channels = 1 if audio.ndim == 1 else audio.shape[0]
            
            # Filter by duration
            if duration < self.min_audio_length or duration > self.max_audio_length:
                logger.debug(f"Audio duration {duration:.2f}s outside range [{self.min_audio_length}, {self.max_audio_length}]")
                return None, None
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                if TORCH_AVAILABLE:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                    audio = resampler(torch.tensor(audio)).numpy()
                else:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sample_rate)
                sample_rate = self.target_sample_rate
            
            # Normalize audio
            audio = self.audio_processor.normalize_audio(audio)
            
            # Save processed audio
            if TORCH_AVAILABLE:
                torchaudio.save(str(output_path), torch.tensor(audio).unsqueeze(0), sample_rate)
            else:
                import soundfile as sf
                sf.write(str(output_path), audio, sample_rate)
            
            audio_info = {
                'duration': len(audio) / sample_rate,
                'sample_rate': sample_rate,
                'channels': channels
            }
            
            return audio, audio_info
            
        except Exception as e:
            logger.error(f"Failed to process audio {input_path}: {e}")
            return None, None
    
    def _create_spectrogram(self, audio: np.ndarray, output_path: Path):
        """Create and save spectrogram."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Create mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=getattr(self.config.audio, 'n_fft', 1024),
                hop_length=getattr(self.config.audio, 'hop_length', 256),
                n_mels=getattr(self.config.audio, 'n_mels', 80)
            )
            
            mel_spec = mel_transform(audio_tensor)
            
            # Convert to log scale
            log_mel_spec = torch.log(mel_spec + 1e-8)
            
            # Save spectrogram
            torch.save(log_mel_spec, output_path)
            
        except Exception as e:
            logger.error(f"Failed to create spectrogram: {e}")
    
    def _create_splits(self, samples: List[Dict], output_path: Path):
        """Create train/validation splits."""
        # Shuffle samples
        import random
        random.shuffle(samples)
        
        # Split ratios
        train_ratio = getattr(self.config.data, 'train_ratio', 0.8)
        val_ratio = getattr(self.config.data, 'val_ratio', 0.1)
        test_ratio = getattr(self.config.data, 'test_ratio', 0.1)
        
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            if split_samples:
                with open(output_path / f"{split_name}.json", 'w') as f:
                    json.dump(split_samples, f, indent=2)
        
        logger.info(f"Created splits: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")


def run(args, config: DictConfig):
    """Run data preprocessing."""
    logger.info("Starting data preprocessing")
    
    # Check required arguments
    if not hasattr(args, 'input_dir') or not args.input_dir:
        logger.error("Input directory not specified. Use --input-dir")
        return 1
    
    if not hasattr(args, 'output_dir') or not args.output_dir:
        logger.error("Output directory not specified. Use --output-dir")
        return 1
    
    # Create preprocessor
    preprocessor = DatasetPreprocessor(config)
    
    # Run preprocessing
    try:
        dataset_format = getattr(args, 'format', 'auto')
        
        if dataset_format == 'ljspeech' or (dataset_format == 'auto' and 'ljspeech' in args.input_dir.lower()):
            result = preprocessor.preprocess_ljspeech(args.input_dir, args.output_dir)
        else:
            result = preprocessor.preprocess_custom_dataset(
                args.input_dir, args.output_dir, dataset_format
            )
        
        # Print statistics
        print("\nPreprocessing Results:")
        print("=" * 50)
        stats = result['statistics']
        print(f"Total samples: {stats['total_samples']}")
        print(f"Successfully processed: {stats['successful_samples']}")
        print(f"Failed: {stats['failed_samples']}")
        print(f"Success rate: {stats['successful_samples']/stats['total_samples']*100:.1f}%")
        
        if stats['audio_stats']['durations']:
            durations = stats['audio_stats']['durations']
            print(f"Audio duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
            print(f"Duration range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
        
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1
    
    logger.info("Preprocessing completed successfully")
    return 0


if __name__ == "__main__":
    # Simple test
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'audio': {
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'n_mels': 80,
            'max_length': 10.0,
            'min_length': 0.5
        },
        'data': {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'text': {
            'language': 'en',
            'phonemize': False
        }
    })
    
    # Create dummy args
    class Args:
        input_dir = "path/to/dataset"
        output_dir = "path/to/output"
        format = "auto"
    
    args = Args()
    print("Preprocessing module loaded successfully")
