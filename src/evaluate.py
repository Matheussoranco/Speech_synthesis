#!/usr/bin/env python3
"""
Model Evaluation Module

Provides comprehensive evaluation functionality for TTS models including:
- Performance metrics (MOS, WER, etc.)
- Audio quality assessment
- Model comparison
- Benchmarking
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import librosa
from omegaconf import DictConfig, OmegaConf

try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not available, some metrics will be skipped")

try:
    import speechmetrics
    SPEECHMETRICS_AVAILABLE = True
except ImportError:
    SPEECHMETRICS_AVAILABLE = False
    warnings.warn("speechmetrics not available, some metrics will be skipped")

from .logging_config import get_logger
from .model import ModelFactory
from .text_processor import TextProcessor
from .utils import get_device, AudioProcessor
from .data import create_dataloader

logger = get_logger(__name__)


class AudioMetrics:
    """Audio quality metrics calculator."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
    def calculate_snr(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        if len(reference) != len(degraded):
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]
        
        noise = reference - degraded
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_pesq(self, reference: np.ndarray, degraded: np.ndarray) -> Optional[float]:
        """Calculate PESQ score."""
        if not PESQ_AVAILABLE:
            return None
            
        try:
            # Ensure proper length and range
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]
            
            # Normalize to [-1, 1] range
            reference = np.clip(reference, -1, 1)
            degraded = np.clip(degraded, -1, 1)
            
            score = pesq.pesq(self.sample_rate, reference, degraded, 'wb')
            return float(score)
        except Exception as e:
            logger.warning(f"Failed to calculate PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference: np.ndarray, degraded: np.ndarray) -> Optional[float]:
        """Calculate Short-Time Objective Intelligibility."""
        try:
            from pystoi import stoi
            
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]
            
            score = stoi(reference, degraded, self.sample_rate, extended=False)
            return float(score)
        except ImportError:
            logger.warning("pystoi not available for STOI calculation")
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate STOI: {e}")
            return None
    
    def calculate_spectral_metrics(self, reference: np.ndarray, degraded: np.ndarray) -> Dict[str, float]:
        """Calculate spectral domain metrics."""
        # Convert to spectrograms
        ref_spec = librosa.stft(reference)
        deg_spec = librosa.stft(degraded)
        
        # Ensure same shape
        min_time = min(ref_spec.shape[1], deg_spec.shape[1])
        ref_spec = ref_spec[:, :min_time]
        deg_spec = deg_spec[:, :min_time]
        
        # Magnitude spectrograms
        ref_mag = np.abs(ref_spec)
        deg_mag = np.abs(deg_spec)
        
        # Spectral distortion
        spec_dist = np.mean((ref_mag - deg_mag) ** 2)
        
        # Log spectral distortion
        ref_log = np.log(ref_mag + 1e-8)
        deg_log = np.log(deg_mag + 1e-8)
        log_spec_dist = np.mean((ref_log - deg_log) ** 2)
        
        return {
            'spectral_distortion': float(spec_dist),
            'log_spectral_distortion': float(log_spec_dist)
        }


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = get_device(config.system.device)
        self.text_processor = TextProcessor(config)
        self.audio_metrics = AudioMetrics(config.audio.sample_rate)
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the model for evaluation."""
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.config.model.type)
        
        # Load checkpoint if specified
        if hasattr(self.config.evaluation, 'checkpoint_path') and self.config.evaluation.checkpoint_path:
            checkpoint_path = Path(self.config.evaluation.checkpoint_path)
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def evaluate_dataset(self, dataset_path: str, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset_path,
            self.config,
            batch_size=1,  # Evaluate one sample at a time
            shuffle=False
        )
        
        results = {
            'audio_metrics': [],
            'timing_metrics': [],
            'error_samples': [],
            'summary': {}
        }
        
        total_inference_time = 0
        successful_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # Extract batch data
                    text = batch.get('text', [''])[0]
                    reference_audio = batch.get('audio', None)
                    
                    if reference_audio is None:
                        continue
                    
                    # Synthesize audio
                    start_time = time.time()
                    synthesized_audio = self._synthesize_audio(text)
                    inference_time = time.time() - start_time
                    
                    if synthesized_audio is None:
                        results['error_samples'].append({
                            'batch_idx': batch_idx,
                            'text': text,
                            'error': 'Synthesis failed'
                        })
                        continue
                    
                    # Calculate audio metrics
                    ref_audio_np = reference_audio.squeeze().cpu().numpy()
                    synth_audio_np = synthesized_audio
                    
                    sample_metrics = self._calculate_sample_metrics(
                        ref_audio_np, synth_audio_np, text, inference_time
                    )
                    
                    results['audio_metrics'].append(sample_metrics)
                    results['timing_metrics'].append({
                        'inference_time': inference_time,
                        'audio_length': len(synth_audio_np) / self.config.audio.sample_rate,
                        'rtf': inference_time / (len(synth_audio_np) / self.config.audio.sample_rate)
                    })
                    
                    total_inference_time += inference_time
                    successful_samples += 1
                    
                    # Save sample results
                    if batch_idx < 10:  # Save first 10 samples
                        sample_dir = output_dir / f"sample_{batch_idx:04d}"
                        sample_dir.mkdir(exist_ok=True)
                        
                        # Save audio files
                        torchaudio.save(
                            sample_dir / "reference.wav",
                            torch.tensor(ref_audio_np).unsqueeze(0),
                            self.config.audio.sample_rate
                        )
                        torchaudio.save(
                            sample_dir / "synthesized.wav",
                            torch.tensor(synth_audio_np).unsqueeze(0),
                            self.config.audio.sample_rate
                        )
                        
                        # Save metrics
                        with open(sample_dir / "metrics.json", 'w') as f:
                            json.dump(sample_metrics, f, indent=2)
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    results['error_samples'].append({
                        'batch_idx': batch_idx,
                        'text': text if 'text' in locals() else 'Unknown',
                        'error': str(e)
                    })
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(
            results['audio_metrics'], results['timing_metrics'], successful_samples
        )
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation completed. Results saved to: {results_file}")
        return results
    
    def _synthesize_audio(self, text: str) -> Optional[np.ndarray]:
        """Synthesize audio from text."""
        try:
            # Process text
            processed_text = self.text_processor.process(text)
            
            # Convert to tensor
            if isinstance(processed_text, str):
                # Simple tokenization if text processor returns string
                tokens = torch.tensor([ord(c) for c in processed_text[:100]], dtype=torch.long)
            else:
                tokens = torch.tensor(processed_text, dtype=torch.long)
            
            tokens = tokens.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Generate audio
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    audio = self.model.generate(tokens)
                elif hasattr(self.model, 'forward'):
                    audio = self.model(tokens)
                else:
                    raise ValueError("Model has no generate or forward method")
                
                if isinstance(audio, tuple):
                    audio = audio[0]  # Take first element if tuple
                
                # Convert to numpy
                if isinstance(audio, torch.Tensor):
                    audio = audio.squeeze().cpu().numpy()
                
                return audio
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None
    
    def _calculate_sample_metrics(self, reference: np.ndarray, synthesized: np.ndarray, 
                                text: str, inference_time: float) -> Dict[str, Any]:
        """Calculate metrics for a single sample."""
        metrics = {
            'text': text,
            'inference_time': inference_time,
            'audio_length': len(synthesized) / self.config.audio.sample_rate
        }
        
        # Basic audio metrics
        metrics['snr'] = self.audio_metrics.calculate_snr(reference, synthesized)
        
        # Advanced metrics if available
        pesq_score = self.audio_metrics.calculate_pesq(reference, synthesized)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score
        
        stoi_score = self.audio_metrics.calculate_stoi(reference, synthesized)
        if stoi_score is not None:
            metrics['stoi'] = stoi_score
        
        # Spectral metrics
        spectral_metrics = self.audio_metrics.calculate_spectral_metrics(reference, synthesized)
        metrics.update(spectral_metrics)
        
        return metrics
    
    def _calculate_summary_statistics(self, audio_metrics: List[Dict], 
                                    timing_metrics: List[Dict], 
                                    successful_samples: int) -> Dict[str, Any]:
        """Calculate summary statistics from all samples."""
        if not audio_metrics:
            return {'error': 'No successful samples'}
        
        summary = {
            'total_samples': successful_samples,
            'mean_inference_time': np.mean([m['inference_time'] for m in timing_metrics]),
            'mean_rtf': np.mean([m['rtf'] for m in timing_metrics]),
            'total_audio_duration': sum(m['audio_length'] for m in timing_metrics)
        }
        
        # Calculate mean metrics
        metric_names = ['snr', 'pesq', 'stoi', 'spectral_distortion', 'log_spectral_distortion']
        
        for metric_name in metric_names:
            values = [m.get(metric_name) for m in audio_metrics if m.get(metric_name) is not None]
            if values:
                summary[f'mean_{metric_name}'] = np.mean(values)
                summary[f'std_{metric_name}'] = np.std(values)
                summary[f'min_{metric_name}'] = np.min(values)
                summary[f'max_{metric_name}'] = np.max(values)
        
        return summary
    
    def benchmark_model(self, texts: List[str], repetitions: int = 10) -> Dict[str, Any]:
        """Benchmark model performance with given texts."""
        logger.info(f"Benchmarking model with {len(texts)} texts, {repetitions} repetitions each")
        
        results = {
            'texts': texts,
            'repetitions': repetitions,
            'measurements': [],
            'summary': {}
        }
        
        for text_idx, text in enumerate(texts):
            text_results = []
            
            for rep in range(repetitions):
                start_time = time.time()
                audio = self._synthesize_audio(text)
                inference_time = time.time() - start_time
                
                if audio is not None:
                    audio_length = len(audio) / self.config.audio.sample_rate
                    rtf = inference_time / audio_length if audio_length > 0 else float('inf')
                    
                    text_results.append({
                        'inference_time': inference_time,
                        'audio_length': audio_length,
                        'rtf': rtf,
                        'success': True
                    })
                else:
                    text_results.append({
                        'inference_time': inference_time,
                        'success': False
                    })
            
            results['measurements'].append({
                'text': text,
                'text_index': text_idx,
                'results': text_results
            })
        
        # Calculate summary
        all_successful = []
        for measurement in results['measurements']:
            successful = [r for r in measurement['results'] if r['success']]
            all_successful.extend(successful)
        
        if all_successful:
            results['summary'] = {
                'success_rate': len(all_successful) / (len(texts) * repetitions),
                'mean_inference_time': np.mean([r['inference_time'] for r in all_successful]),
                'std_inference_time': np.std([r['inference_time'] for r in all_successful]),
                'mean_rtf': np.mean([r['rtf'] for r in all_successful]),
                'std_rtf': np.std([r['rtf'] for r in all_successful])
            }
        
        return results


def run(args, config: DictConfig):
    """Run model evaluation."""
    logger.info("Starting model evaluation")
    
    evaluator = ModelEvaluator(config)
    
    # Determine evaluation type
    if hasattr(args, 'dataset') and args.dataset:
        # Dataset evaluation
        results = evaluator.evaluate_dataset(
            args.dataset,
            args.output_dir if hasattr(args, 'output_dir') else "evaluation_results"
        )
        
        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 50)
        for key, value in results['summary'].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    elif hasattr(args, 'benchmark') and args.benchmark:
        # Benchmark evaluation
        benchmark_texts = [
            "Hello, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis is the artificial production of human speech.",
            "This is a longer sentence to test the model's performance on extended text sequences."
        ]
        
        results = evaluator.benchmark_model(
            benchmark_texts,
            args.repetitions if hasattr(args, 'repetitions') else 10
        )
        
        # Print benchmark results
        print("\nBenchmark Results:")
        print("=" * 50)
        for key, value in results['summary'].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    else:
        logger.error("No evaluation type specified. Use --dataset or --benchmark")
        return 1
    
    logger.info("Evaluation completed successfully")
    return 0


if __name__ == "__main__":
    # Simple test
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'audio': {'sample_rate': 22050},
        'system': {'device': 'auto'},
        'evaluation': {'checkpoint_path': None}
    })
    
    # Create dummy args
    class Args:
        benchmark = True
        repetitions = 3
    
    args = Args()
    run(args, config)
