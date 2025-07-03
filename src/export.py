#!/usr/bin/env python3
"""
Model Export Module

Provides comprehensive model export functionality for production deployment:
- ONNX export for cross-platform inference
- TorchScript export for optimized PyTorch inference
- Mobile optimization
- Quantization support
- Deployment packages
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    import torch
    import torch.nn as nn
    from omegaconf import DictConfig, OmegaConf
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, export functionality limited")

from .logging_config import get_logger
from .model import ModelFactory
from .text_processor import TextProcessor
from .utils import get_device

logger = get_logger()


class ModelExporter:
    """Comprehensive model export system."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = get_device(config.system.device)
        self.text_processor = TextProcessor(config)
        
    def export_onnx(self, model_path: str, output_path: str, 
                   example_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Export model to ONNX format."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for ONNX export")
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Load model
        model = self._load_model(model_path)
        model.eval()
        
        # Prepare example input
        if example_input is None:
            example_input = self._create_example_input()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                example_input,
                str(output_path),
                input_names=['input_ids'],
                output_names=['audio_output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'audio_output': {0: 'batch_size', 1: 'audio_length'}
                },
                opset_version=11,
                do_constant_folding=True,
                verbose=False
            )
            
            # Verify ONNX model
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Get model info
            model_info = {
                'input_shape': list(example_input.shape),
                'opset_version': 11,
                'file_size': output_path.stat().st_size,
                'export_success': True
            }
            
            logger.info(f"ONNX export successful: {output_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return {'export_success': False, 'error': str(e)}
    
    def export_torchscript(self, model_path: str, output_path: str,
                          method: str = "trace") -> Dict[str, Any]:
        """Export model to TorchScript format."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for TorchScript export")
        
        logger.info(f"Exporting model to TorchScript ({method}): {output_path}")
        
        # Load model
        model = self._load_model(model_path)
        model.eval()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if method == "trace":
                # Trace the model
                example_input = self._create_example_input()
                traced_model = torch.jit.trace(model, example_input)
                scripted_model = traced_model
            elif method == "script":
                # Script the model
                scripted_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown export method: {method}")
            
            # Save the model
            scripted_model.save(str(output_path))
            
            # Test the exported model
            test_input = self._create_example_input()
            with torch.no_grad():
                original_output = model(test_input)
                scripted_output = scripted_model(test_input)
                
                # Compare outputs
                if isinstance(original_output, tuple):
                    original_output = original_output[0]
                if isinstance(scripted_output, tuple):
                    scripted_output = scripted_output[0]
                
                max_diff = torch.max(torch.abs(original_output - scripted_output)).item()
            
            model_info = {
                'method': method,
                'file_size': output_path.stat().st_size,
                'max_output_diff': max_diff,
                'export_success': True
            }
            
            logger.info(f"TorchScript export successful: {output_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return {'export_success': False, 'error': str(e)}
    
    def export_mobile(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """Export model optimized for mobile deployment."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for mobile export")
        
        logger.info(f"Exporting model for mobile: {output_path}")
        
        # Load model
        model = self._load_model(model_path)
        model.eval()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert to TorchScript
            example_input = self._create_example_input()
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize for mobile
            optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            # Save mobile model
            optimized_model._save_for_lite_interpreter(str(output_path))
            
            model_info = {
                'file_size': output_path.stat().st_size,
                'optimization': 'mobile',
                'export_success': True
            }
            
            logger.info(f"Mobile export successful: {output_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"Mobile export failed: {e}")
            return {'export_success': False, 'error': str(e)}
    
    def quantize_model(self, model_path: str, output_path: str,
                      quantization_type: str = "dynamic") -> Dict[str, Any]:
        """Quantize model for faster inference."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for quantization")
        
        logger.info(f"Quantizing model ({quantization_type}): {output_path}")
        
        # Load model
        model = self._load_model(model_path)
        model.eval()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv1d},
                    dtype=torch.qint8
                )
            elif quantization_type == "static":
                # Static quantization (requires calibration dataset)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibrate with example data
                example_input = self._create_example_input()
                with torch.no_grad():
                    model(example_input)
                
                quantized_model = torch.quantization.convert(model, inplace=False)
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), output_path)
            
            # Compare model sizes and performance
            original_size = sum(p.numel() * p.element_size() for p in model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            
            # Test inference speed
            import time
            test_input = self._create_example_input()
            
            # Original model timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    model(test_input)
            original_time = (time.time() - start_time) / 10
            
            # Quantized model timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    quantized_model(test_input)
            quantized_time = (time.time() - start_time) / 10
            
            model_info = {
                'quantization_type': quantization_type,
                'original_size_mb': original_size / (1024 * 1024),
                'quantized_size_mb': quantized_size / (1024 * 1024),
                'size_reduction': (original_size - quantized_size) / original_size,
                'original_inference_time': original_time,
                'quantized_inference_time': quantized_time,
                'speedup': original_time / quantized_time if quantized_time > 0 else 0,
                'export_success': True
            }
            
            logger.info(f"Quantization successful: {output_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return {'export_success': False, 'error': str(e)}
    
    def create_deployment_package(self, model_path: str, output_dir: str,
                                include_formats: List[str] = None) -> Dict[str, Any]:
        """Create a complete deployment package."""
        logger.info(f"Creating deployment package: {output_dir}")
        
        if include_formats is None:
            include_formats = ['torchscript', 'onnx']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        package_info = {
            'model_exports': {},
            'package_contents': [],
            'creation_success': True
        }
        
        try:
            # Export in different formats
            for format_name in include_formats:
                if format_name == 'torchscript':
                    export_path = output_path / "model.pt"
                    result = self.export_torchscript(model_path, str(export_path))
                    package_info['model_exports']['torchscript'] = result
                
                elif format_name == 'onnx':
                    export_path = output_path / "model.onnx"
                    result = self.export_onnx(model_path, str(export_path))
                    package_info['model_exports']['onnx'] = result
                
                elif format_name == 'mobile':
                    export_path = output_path / "model_mobile.ptl"
                    result = self.export_mobile(model_path, str(export_path))
                    package_info['model_exports']['mobile'] = result
                
                elif format_name == 'quantized':
                    export_path = output_path / "model_quantized.pt"
                    result = self.quantize_model(model_path, str(export_path))
                    package_info['model_exports']['quantized'] = result
            
            # Copy configuration
            config_path = output_path / "config.yaml"
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(self.config))
            package_info['package_contents'].append('config.yaml')
            
            # Create inference script
            inference_script = self._create_inference_script()
            script_path = output_path / "inference.py"
            with open(script_path, 'w') as f:
                f.write(inference_script)
            package_info['package_contents'].append('inference.py')
            
            # Create requirements file
            requirements = self._create_requirements()
            req_path = output_path / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write('\n'.join(requirements))
            package_info['package_contents'].append('requirements.txt')
            
            # Create README
            readme = self._create_deployment_readme()
            readme_path = output_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme)
            package_info['package_contents'].append('README.md')
            
            # Save package info
            info_path = output_path / "package_info.json"
            with open(info_path, 'w') as f:
                json.dump(package_info, f, indent=2, default=str)
            package_info['package_contents'].append('package_info.json')
            
            logger.info(f"Deployment package created: {output_path}")
            return package_info
            
        except Exception as e:
            logger.error(f"Package creation failed: {e}")
            package_info['creation_success'] = False
            package_info['error'] = str(e)
            return package_info
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.config.model.type)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def _create_example_input(self) -> torch.Tensor:
        """Create example input for model export."""
        # Create dummy text input
        batch_size = 1
        seq_length = 50  # Example sequence length
        
        # Simple tokenization for export
        example_input = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)
        return example_input.to(self.device)
    
    def _create_inference_script(self) -> str:
        """Create inference script for deployment."""
        script = '''#!/usr/bin/env python3
"""
Speech Synthesis Inference Script

Simple inference script for deployed TTS model.
Supports multiple model formats (PyTorch, ONNX, TorchScript).
"""

import argparse
import torch
import torchaudio
from pathlib import Path

def load_torchscript_model(model_path):
    """Load TorchScript model."""
    return torch.jit.load(model_path)

def load_onnx_model(model_path):
    """Load ONNX model."""
    import onnxruntime as ort
    return ort.InferenceSession(model_path)

def synthesize_speech(model, text, model_type="torchscript"):
    """Synthesize speech from text."""
    # Simple tokenization (replace with proper text processing)
    tokens = torch.tensor([ord(c) for c in text[:50]], dtype=torch.long).unsqueeze(0)
    
    if model_type == "torchscript":
        with torch.no_grad():
            audio = model(tokens)
    elif model_type == "onnx":
        inputs = {model.get_inputs()[0].name: tokens.numpy()}
        audio = model.run(None, inputs)[0]
        audio = torch.tensor(audio)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return audio.squeeze()

def main():
    parser = argparse.ArgumentParser(description="TTS Inference")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--format", default="torchscript", 
                       choices=["torchscript", "onnx"], help="Model format")
    
    args = parser.parse_args()
    
    # Load model
    if args.format == "torchscript":
        model = load_torchscript_model(args.model)
    elif args.format == "onnx":
        model = load_onnx_model(args.model)
    
    # Synthesize
    audio = synthesize_speech(model, args.text, args.format)
    
    # Save audio
    torchaudio.save(args.output, audio.unsqueeze(0), 22050)
    print(f"Audio saved to: {args.output}")

if __name__ == "__main__":
    main()
'''
        return script
    
    def _create_requirements(self) -> List[str]:
        """Create requirements for deployment."""
        requirements = [
            "torch>=1.9.0",
            "torchaudio>=0.9.0",
            "numpy>=1.21.0"
        ]
        
        # Add optional dependencies based on export formats
        if 'onnx' in str(self.config):
            requirements.extend([
                "onnx>=1.10.0",
                "onnxruntime>=1.9.0"
            ])
        
        return requirements
    
    def _create_deployment_readme(self) -> str:
        """Create README for deployment package."""
        readme = '''# Speech Synthesis Deployment Package

This package contains exported models and utilities for speech synthesis inference.

## Contents

- `model.pt` - TorchScript model (if available)
- `model.onnx` - ONNX model (if available)
- `model_mobile.ptl` - Mobile-optimized model (if available)
- `model_quantized.pt` - Quantized model (if available)
- `inference.py` - Simple inference script
- `config.yaml` - Model configuration
- `requirements.txt` - Python dependencies

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run inference:
   ```bash
   python inference.py --model model.pt --text "Hello world" --output output.wav
   ```

## Model Formats

### TorchScript (.pt)
- Native PyTorch format
- Good performance, full feature support
- Use with: `torch.jit.load()`

### ONNX (.onnx)
- Cross-platform format
- Compatible with many inference engines
- Use with: `onnxruntime.InferenceSession()`

### Mobile (.ptl)
- Optimized for mobile deployment
- Smaller size, faster inference
- Use with PyTorch Mobile

### Quantized (.pt)
- Reduced precision for efficiency
- Smaller model size, faster inference
- Some accuracy trade-off

## Integration

For production integration, adapt the inference script to your needs:
- Add proper text preprocessing
- Handle batch processing
- Add error handling
- Implement audio post-processing

## Performance Tips

1. Use appropriate model format for your deployment target
2. Consider quantization for resource-constrained environments
3. Batch multiple requests for better throughput
4. Cache models in memory for repeated inference

## Support

For issues and questions, refer to the main project documentation.
'''
        return readme


def run(args, config: DictConfig):
    """Run model export."""
    logger.info("Starting model export")
    
    # Check required arguments
    if not hasattr(args, 'model_path') or not args.model_path:
        logger.error("Model path not specified. Use --model-path")
        return 1
    
    if not hasattr(args, 'output') or not args.output:
        logger.error("Output path not specified. Use --output")
        return 1
    
    # Create exporter
    exporter = ModelExporter(config)
    
    # Determine export type
    export_format = getattr(args, 'format', 'torchscript')
    
    try:
        if export_format == 'package':
            # Create deployment package
            formats = getattr(args, 'include_formats', ['torchscript', 'onnx'])
            if isinstance(formats, str):
                formats = formats.split(',')
            
            result = exporter.create_deployment_package(
                args.model_path, args.output, formats
            )
            
            if result['creation_success']:
                print(f"Deployment package created: {args.output}")
                print(f"Included formats: {list(result['model_exports'].keys())}")
            else:
                print(f"Package creation failed: {result.get('error', 'Unknown error')}")
                return 1
        
        elif export_format == 'torchscript':
            result = exporter.export_torchscript(args.model_path, args.output)
            
        elif export_format == 'onnx':
            result = exporter.export_onnx(args.model_path, args.output)
            
        elif export_format == 'mobile':
            result = exporter.export_mobile(args.model_path, args.output)
            
        elif export_format == 'quantized':
            quant_type = getattr(args, 'quantization_type', 'dynamic')
            result = exporter.quantize_model(args.model_path, args.output, quant_type)
            
        else:
            logger.error(f"Unknown export format: {export_format}")
            return 1
        
        # Print results for single format exports
        if export_format != 'package':
            if result.get('export_success', False):
                print(f"Export successful: {args.output}")
                if 'file_size' in result:
                    print(f"File size: {result['file_size'] / (1024*1024):.1f} MB")
                if 'speedup' in result:
                    print(f"Speedup: {result['speedup']:.2f}x")
            else:
                print(f"Export failed: {result.get('error', 'Unknown error')}")
                return 1
    
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    
    logger.info("Export completed successfully")
    return 0


if __name__ == "__main__":
    # Simple test
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'system': {'device': 'auto'}
    })
    
    print("Export module loaded successfully")
