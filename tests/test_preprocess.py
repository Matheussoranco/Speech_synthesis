#!/usr/bin/env python3
"""
Tests for preprocessing module.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

import numpy as np
from omegaconf import OmegaConf

from src.preprocess import DatasetPreprocessor, run


class TestDatasetPreprocessor:
    """Test dataset preprocessing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OmegaConf.create({
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
        
        self.preprocessor = DatasetPreprocessor(self.config)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor.config == self.config
        assert self.preprocessor.target_sample_rate == 22050
        assert self.preprocessor.max_audio_length == 10.0
        assert self.preprocessor.min_audio_length == 0.5
    
    def test_detect_ljspeech_format(self):
        """Test LJSpeech format detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create metadata.csv file
            (temp_path / "metadata.csv").touch()
            
            format_detected = self.preprocessor._detect_dataset_format(temp_path)
            assert format_detected == "ljspeech"
    
    def test_detect_common_voice_format(self):
        """Test Common Voice format detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create validated.tsv file
            (temp_path / "validated.tsv").touch()
            
            format_detected = self.preprocessor._detect_dataset_format(temp_path)
            assert format_detected == "common_voice"
    
    def test_detect_generic_format(self):
        """Test generic format detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            format_detected = self.preprocessor._detect_dataset_format(temp_path)
            assert format_detected == "generic"
    
    @patch('src.preprocess.torchaudio.load')
    @patch('src.preprocess.torchaudio.save')
    @patch('src.preprocess.TORCH_AVAILABLE', True)
    def test_process_audio_file_success(self, mock_save, mock_load):
        """Test successful audio file processing."""
        # Mock audio data
        sample_rate = 44100
        duration = 2.0
        n_samples = int(sample_rate * duration)
        mock_waveform = torch.randn(1, n_samples)
        
        mock_load.return_value = (mock_waveform, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            
            audio, info = self.preprocessor._process_audio_file(input_path, output_path)
            
            assert audio is not None
            assert info is not None
            assert 'duration' in info
            assert 'sample_rate' in info
            assert 'channels' in info
            
            mock_load.assert_called_once()
            mock_save.assert_called_once()
    
    @patch('src.preprocess.torchaudio.load')
    @patch('src.preprocess.TORCH_AVAILABLE', True)
    def test_process_audio_file_too_short(self, mock_load):
        """Test audio file that's too short."""
        # Mock very short audio
        sample_rate = 22050
        duration = 0.1  # Too short
        n_samples = int(sample_rate * duration)
        mock_waveform = torch.randn(1, n_samples)
        
        mock_load.return_value = (mock_waveform, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            
            audio, info = self.preprocessor._process_audio_file(input_path, output_path)
            
            assert audio is None
            assert info is None
    
    @patch('src.preprocess.torchaudio.load')
    @patch('src.preprocess.TORCH_AVAILABLE', True)
    def test_process_audio_file_too_long(self, mock_load):
        """Test audio file that's too long."""
        # Mock very long audio
        sample_rate = 22050
        duration = 15.0  # Too long
        n_samples = int(sample_rate * duration)
        mock_waveform = torch.randn(1, n_samples)
        
        mock_load.return_value = (mock_waveform, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            
            audio, info = self.preprocessor._process_audio_file(input_path, output_path)
            
            assert audio is None
            assert info is None
    
    def test_create_splits(self):
        """Test dataset splitting."""
        # Create sample data
        samples = [
            {'id': f'sample_{i}', 'text': f'Text {i}', 'duration': 1.0}
            for i in range(100)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            self.preprocessor._create_splits(samples, output_path)
            
            # Check that split files are created
            assert (output_path / "train.json").exists()
            assert (output_path / "validation.json").exists()
            assert (output_path / "test.json").exists()
            
            # Check split proportions
            with open(output_path / "train.json", 'r') as f:
                train_data = json.load(f)
            with open(output_path / "validation.json", 'r') as f:
                val_data = json.load(f)
            with open(output_path / "test.json", 'r') as f:
                test_data = json.load(f)
            
            total_samples = len(train_data) + len(val_data) + len(test_data)
            assert total_samples <= len(samples)  # Some might be lost due to rounding
            
            # Check approximate ratios
            train_ratio = len(train_data) / total_samples
            val_ratio = len(val_data) / total_samples
            
            assert 0.75 <= train_ratio <= 0.85  # Approximately 0.8
            assert 0.05 <= val_ratio <= 0.15   # Approximately 0.1
    
    @patch('builtins.open', new_callable=mock_open, read_data="LJ001-0001|text1|Normalized text 1\nLJ001-0002|text2|Normalized text 2\n")
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_preprocess_ljspeech_metadata_parsing(self, mock_mkdir, mock_exists, mock_file):
        """Test LJSpeech metadata parsing."""
        mock_exists.return_value = True
        
        with patch.object(self.preprocessor, '_process_audio_file') as mock_process_audio:
            with patch.object(self.preprocessor, '_create_spectrogram'):
                with patch.object(self.preprocessor, '_create_splits'):
                    # Mock successful audio processing
                    mock_process_audio.return_value = (
                        np.random.randn(22050),  # 1 second of audio
                        {'duration': 1.0, 'sample_rate': 22050, 'channels': 1}
                    )
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_dir = Path(temp_dir) / "input"
                        output_dir = Path(temp_dir) / "output"
                        
                        result = self.preprocessor.preprocess_ljspeech(str(input_dir), str(output_dir))
                        
                        assert 'dataset_type' in result
                        assert result['dataset_type'] == 'ljspeech'
                        assert 'samples' in result
                        assert 'statistics' in result
    
    def test_preprocess_generic_with_json(self):
        """Test generic preprocessing with JSON metadata."""
        json_data = [
            {'audio': 'audio1.wav', 'text': 'Hello world', 'id': 'sample1'},
            {'audio': 'audio2.wav', 'text': 'Test sample', 'id': 'sample2'}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            output_path = Path(temp_dir) / "output"
            input_path.mkdir()
            
            # Create JSON metadata file
            with open(input_path / "metadata.json", 'w') as f:
                json.dump(json_data, f)
            
            # Create dummy audio files
            (input_path / "audio1.wav").touch()
            (input_path / "audio2.wav").touch()
            
            with patch.object(self.preprocessor, '_process_samples') as mock_process:
                mock_process.return_value = {'dataset_type': 'generic', 'samples': []}
                
                result = self.preprocessor._preprocess_generic(input_path, output_path)
                
                mock_process.assert_called_once()
                args, kwargs = mock_process.call_args
                samples = args[0]
                
                assert len(samples) == 2
                assert samples[0]['id'] == 'sample1'
                assert samples[1]['id'] == 'sample2'


def test_run_preprocessing_success():
    """Test successful preprocessing run."""
    config = OmegaConf.create({
        'audio': {'sample_rate': 22050},
        'data': {'train_ratio': 0.8}
    })
    
    args = Mock()
    args.input_dir = "input_path"
    args.output_dir = "output_path"
    args.format = "ljspeech"
    
    with patch('src.preprocess.DatasetPreprocessor') as mock_preprocessor_class:
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_ljspeech.return_value = {
            'statistics': {
                'total_samples': 100,
                'successful_samples': 95,
                'failed_samples': 5,
                'audio_stats': {'durations': [1.0, 2.0, 1.5]}
            }
        }
        mock_preprocessor_class.return_value = mock_preprocessor
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_preprocessor.preprocess_ljspeech.assert_called_once_with("input_path", "output_path")


def test_run_preprocessing_missing_input():
    """Test preprocessing with missing input directory."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.input_dir = None
    
    result = run(args, config)
    
    assert result == 1  # Error


def test_run_preprocessing_missing_output():
    """Test preprocessing with missing output directory."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.input_dir = "input_path"
    args.output_dir = None
    
    result = run(args, config)
    
    assert result == 1  # Error


def test_run_preprocessing_custom_format():
    """Test preprocessing with custom format."""
    config = OmegaConf.create({
        'audio': {'sample_rate': 22050},
        'data': {'train_ratio': 0.8}
    })
    
    args = Mock()
    args.input_dir = "input_path"
    args.output_dir = "output_path"
    args.format = "vctk"
    
    with patch('src.preprocess.DatasetPreprocessor') as mock_preprocessor_class:
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_custom_dataset.return_value = {
            'statistics': {
                'total_samples': 50,
                'successful_samples': 48,
                'failed_samples': 2,
                'audio_stats': {'durations': [0.8, 1.2, 2.1]}
            }
        }
        mock_preprocessor_class.return_value = mock_preprocessor
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_preprocessor.preprocess_custom_dataset.assert_called_once_with(
            "input_path", "output_path", "vctk"
        )


def test_run_preprocessing_exception():
    """Test preprocessing with exception."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.input_dir = "input_path"
    args.output_dir = "output_path"
    args.format = "auto"
    
    with patch('src.preprocess.DatasetPreprocessor') as mock_preprocessor_class:
        mock_preprocessor_class.side_effect = RuntimeError("Processing error")
        
        result = run(args, config)
        
        assert result == 1  # Error


if __name__ == "__main__":
    pytest.main([__file__])
