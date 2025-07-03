#!/usr/bin/env python3
"""
Tests for evaluation module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np
from omegaconf import OmegaConf

from src.evaluate import AudioMetrics, ModelEvaluator, run


class TestAudioMetrics:
    """Test audio metrics calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = AudioMetrics(sample_rate=22050)
        
        # Create test audio signals
        self.sample_rate = 22050
        duration = 1.0  # 1 second
        self.n_samples = int(duration * self.sample_rate)
        
        # Reference signal (sine wave)
        t = np.linspace(0, duration, self.n_samples)
        self.reference = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Degraded signal (with noise)
        noise = np.random.normal(0, 0.1, self.n_samples)
        self.degraded = self.reference + noise
    
    def test_calculate_snr(self):
        """Test SNR calculation."""
        snr = self.metrics.calculate_snr(self.reference, self.degraded)
        
        assert isinstance(snr, float)
        assert snr > 0  # Should be positive for reasonable signal/noise ratio
        assert snr < 50  # Shouldn't be too high for noisy signal
    
    def test_calculate_snr_perfect_signal(self):
        """Test SNR with identical signals."""
        snr = self.metrics.calculate_snr(self.reference, self.reference)
        
        assert snr == float('inf')  # Perfect signal should have infinite SNR
    
    def test_calculate_snr_different_lengths(self):
        """Test SNR with different length signals."""
        short_signal = self.reference[:1000]
        snr = self.metrics.calculate_snr(self.reference, short_signal)
        
        assert isinstance(snr, float)
        # Should handle length mismatch gracefully
    
    @patch('src.evaluate.PESQ_AVAILABLE', True)
    @patch('pesq.pesq')
    def test_calculate_pesq_success(self, mock_pesq):
        """Test successful PESQ calculation."""
        mock_pesq.return_value = 2.5
        
        pesq_score = self.metrics.calculate_pesq(self.reference, self.degraded)
        
        assert pesq_score == 2.5
        mock_pesq.assert_called_once()
    
    @patch('src.evaluate.PESQ_AVAILABLE', False)
    def test_calculate_pesq_unavailable(self):
        """Test PESQ when not available."""
        pesq_score = self.metrics.calculate_pesq(self.reference, self.degraded)
        
        assert pesq_score is None
    
    def test_calculate_spectral_metrics(self):
        """Test spectral metrics calculation."""
        metrics = self.metrics.calculate_spectral_metrics(self.reference, self.degraded)
        
        assert isinstance(metrics, dict)
        assert 'spectral_distortion' in metrics
        assert 'log_spectral_distortion' in metrics
        
        assert isinstance(metrics['spectral_distortion'], float)
        assert isinstance(metrics['log_spectral_distortion'], float)
        assert metrics['spectral_distortion'] >= 0
        assert metrics['log_spectral_distortion'] >= 0


class TestModelEvaluator:
    """Test model evaluator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OmegaConf.create({
            'model': {'type': 'advanced'},
            'audio': {'sample_rate': 22050},
            'system': {'device': 'cpu'},
            'evaluation': {'checkpoint_path': None}
        })
    
    @patch('src.evaluate.ModelFactory')
    @patch('src.evaluate.TextProcessor')
    def test_evaluator_initialization(self, mock_text_processor, mock_model_factory):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(self.config)
        
        assert evaluator.config == self.config
        assert evaluator.device == 'cpu'
        mock_text_processor.assert_called_once_with(self.config)
        mock_model_factory.assert_called_once_with(self.config)
    
    @patch('src.evaluate.ModelFactory')
    @patch('src.evaluate.TextProcessor')
    def test_synthesize_audio_success(self, mock_text_processor, mock_model_factory):
        """Test successful audio synthesis."""
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = torch.randn(1, 1000)
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        # Mock text processor
        mock_text_processor.return_value.process.return_value = [1, 2, 3, 4, 5]
        
        evaluator = ModelEvaluator(self.config)
        audio = evaluator._synthesize_audio("test text")
        
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
    
    @patch('src.evaluate.ModelFactory')
    @patch('src.evaluate.TextProcessor')
    def test_synthesize_audio_failure(self, mock_text_processor, mock_model_factory):
        """Test audio synthesis failure."""
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.generate.side_effect = RuntimeError("Model error")
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        # Mock text processor
        mock_text_processor.return_value.process.return_value = [1, 2, 3, 4, 5]
        
        evaluator = ModelEvaluator(self.config)
        audio = evaluator._synthesize_audio("test text")
        
        assert audio is None
    
    @patch('src.evaluate.ModelFactory')
    @patch('src.evaluate.TextProcessor')
    def test_benchmark_model(self, mock_text_processor, mock_model_factory):
        """Test model benchmarking."""
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = torch.randn(1, 1000)
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        # Mock text processor
        mock_text_processor.return_value.process.return_value = [1, 2, 3, 4, 5]
        
        evaluator = ModelEvaluator(self.config)
        
        texts = ["Hello", "World"]
        results = evaluator.benchmark_model(texts, repetitions=2)
        
        assert isinstance(results, dict)
        assert 'texts' in results
        assert 'repetitions' in results
        assert 'measurements' in results
        assert 'summary' in results
        
        assert results['texts'] == texts
        assert results['repetitions'] == 2
        assert len(results['measurements']) == 2  # One per text
        
        # Check summary statistics
        summary = results['summary']
        assert 'success_rate' in summary
        assert 'mean_inference_time' in summary
        assert 'mean_rtf' in summary


def test_run_benchmark_mode():
    """Test run function in benchmark mode."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'audio': {'sample_rate': 22050},
        'system': {'device': 'cpu'},
        'evaluation': {'checkpoint_path': None}
    })
    
    # Create mock args
    args = Mock()
    args.benchmark = True
    args.repetitions = 3
    args.dataset = None
    
    with patch('src.evaluate.ModelEvaluator') as mock_evaluator_class:
        mock_evaluator = Mock()
        mock_evaluator.benchmark_model.return_value = {
            'summary': {
                'success_rate': 1.0,
                'mean_inference_time': 0.1,
                'mean_rtf': 0.05
            }
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_evaluator.benchmark_model.assert_called_once()


def test_run_dataset_mode():
    """Test run function in dataset mode."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'audio': {'sample_rate': 22050},
        'system': {'device': 'cpu'},
        'evaluation': {'checkpoint_path': None}
    })
    
    # Create mock args
    args = Mock()
    args.benchmark = False
    args.dataset = "test_dataset"
    args.output_dir = "test_output"
    
    with patch('src.evaluate.ModelEvaluator') as mock_evaluator_class:
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            'summary': {
                'total_samples': 100,
                'successful_samples': 95,
                'mean_snr': 15.2
            }
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_evaluator.evaluate_dataset.assert_called_once_with("test_dataset", "test_output")


def test_run_no_evaluation_type():
    """Test run function with no evaluation type specified."""
    config = OmegaConf.create({'model': {'type': 'advanced'}})
    
    # Create mock args without benchmark or dataset
    args = Mock()
    args.benchmark = False
    args.dataset = None
    
    result = run(args, config)
    
    assert result == 1  # Error


if __name__ == "__main__":
    pytest.main([__file__])
