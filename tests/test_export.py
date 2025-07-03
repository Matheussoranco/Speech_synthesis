#!/usr/bin/env python3
"""
Tests for export module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
from omegaconf import OmegaConf

from src.export import ModelExporter, run


class TestModelExporter:
    """Test model export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OmegaConf.create({
            'model': {'type': 'advanced'},
            'system': {'device': 'cpu'}
        })
        
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_exporter_initialization(self, mock_text_processor, mock_model_factory):
        """Test exporter initialization."""
        exporter = ModelExporter(self.config)
        
        assert exporter.config == self.config
        assert exporter.device == 'cpu'
        mock_text_processor.assert_called_once_with(self.config)
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.load')
    def test_load_model_with_state_dict(self, mock_torch_load, mock_text_processor, mock_model_factory):
        """Test model loading with state dict in checkpoint."""
        # Mock model and checkpoint
        mock_model = Mock()
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        mock_checkpoint = {'model_state_dict': {'weight': torch.randn(10, 10)}}
        mock_torch_load.return_value = mock_checkpoint
        
        exporter = ModelExporter(self.config)
        model = exporter._load_model("fake_path.pt")
        
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint['model_state_dict'])
        mock_model.to.assert_called_once_with('cpu')
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.load')
    def test_load_model_direct_state_dict(self, mock_torch_load, mock_text_processor, mock_model_factory):
        """Test model loading with direct state dict."""
        # Mock model and checkpoint
        mock_model = Mock()
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        mock_state_dict = {'weight': torch.randn(10, 10)}
        mock_torch_load.return_value = mock_state_dict
        
        exporter = ModelExporter(self.config)
        model = exporter._load_model("fake_path.pt")
        
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict)
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_create_example_input(self, mock_text_processor, mock_model_factory):
        """Test example input creation."""
        exporter = ModelExporter(self.config)
        example_input = exporter._create_example_input()
        
        assert isinstance(example_input, torch.Tensor)
        assert example_input.dtype == torch.long
        assert example_input.shape[0] == 1  # Batch size
        assert example_input.shape[1] == 50  # Sequence length
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.onnx.export')
    @patch('onnx.load')
    @patch('onnx.checker.check_model')
    def test_export_onnx_success(self, mock_check, mock_onnx_load, mock_onnx_export, 
                                mock_text_processor, mock_model_factory):
        """Test successful ONNX export."""
        # Mock model
        mock_model = Mock()
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_path = Path(temp_dir) / "model.onnx"
            
            # Create dummy model file
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                exporter = ModelExporter(self.config)
                result = exporter.export_onnx(str(model_path), str(output_path))
            
            assert result['export_success'] is True
            assert 'input_shape' in result
            assert 'opset_version' in result
            mock_onnx_export.assert_called_once()
            mock_check.assert_called_once()
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.onnx.export')
    def test_export_onnx_failure(self, mock_onnx_export, mock_text_processor, mock_model_factory):
        """Test ONNX export failure."""
        # Mock export failure
        mock_onnx_export.side_effect = RuntimeError("Export failed")
        
        # Mock model
        mock_model = Mock()
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_path = Path(temp_dir) / "model.onnx"
            
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                exporter = ModelExporter(self.config)
                result = exporter.export_onnx(str(model_path), str(output_path))
            
            assert result['export_success'] is False
            assert 'error' in result
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.jit.trace')
    def test_export_torchscript_trace(self, mock_trace, mock_text_processor, mock_model_factory):
        """Test TorchScript export with tracing."""
        # Mock model
        mock_model = Mock()
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        # Mock traced model
        mock_traced = Mock()
        mock_traced.return_value = mock_output
        mock_traced.save = Mock()
        mock_trace.return_value = mock_traced
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_path = Path(temp_dir) / "model_script.pt"
            
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                exporter = ModelExporter(self.config)
                result = exporter.export_torchscript(str(model_path), str(output_path), method="trace")
            
            assert result['export_success'] is True
            assert result['method'] == 'trace'
            assert 'max_output_diff' in result
            mock_trace.assert_called_once()
            mock_traced.save.assert_called_once()
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.jit.script')
    def test_export_torchscript_script(self, mock_script, mock_text_processor, mock_model_factory):
        """Test TorchScript export with scripting."""
        # Mock model
        mock_model = Mock()
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        # Mock scripted model
        mock_scripted = Mock()
        mock_scripted.return_value = mock_output
        mock_scripted.save = Mock()
        mock_script.return_value = mock_scripted
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_path = Path(temp_dir) / "model_script.pt"
            
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                exporter = ModelExporter(self.config)
                result = exporter.export_torchscript(str(model_path), str(output_path), method="script")
            
            assert result['export_success'] is True
            assert result['method'] == 'script'
            mock_script.assert_called_once()
            mock_scripted.save.assert_called_once()
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    @patch('torch.quantization.quantize_dynamic')
    def test_quantize_model_dynamic(self, mock_quantize, mock_text_processor, mock_model_factory):
        """Test dynamic model quantization."""
        # Mock model
        mock_model = Mock()
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        mock_model_factory.return_value.create_model.return_value = mock_model
        mock_model.parameters.return_value = [torch.randn(100, 100)]
        
        # Mock quantized model
        mock_quantized = Mock()
        mock_quantized.return_value = mock_output
        mock_quantized.parameters.return_value = [torch.randn(100, 100)]
        mock_quantize.return_value = mock_quantized
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_path = Path(temp_dir) / "model_quantized.pt"
            
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                with patch('torch.save'):
                    exporter = ModelExporter(self.config)
                    result = exporter.quantize_model(str(model_path), str(output_path), "dynamic")
            
            assert result['export_success'] is True
            assert result['quantization_type'] == 'dynamic'
            assert 'size_reduction' in result
            assert 'speedup' in result
            mock_quantize.assert_called_once()
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_create_deployment_package(self, mock_text_processor, mock_model_factory):
        """Test deployment package creation."""
        mock_model = Mock()
        mock_model_factory.return_value.create_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            output_dir = Path(temp_dir) / "package"
            
            torch.save({'dummy': 'data'}, model_path)
            
            with patch.object(ModelExporter, '_load_model', return_value=mock_model):
                with patch.object(ModelExporter, 'export_torchscript', return_value={'export_success': True}):
                    with patch.object(ModelExporter, 'export_onnx', return_value={'export_success': True}):
                        exporter = ModelExporter(self.config)
                        result = exporter.create_deployment_package(
                            str(model_path), str(output_dir), ['torchscript', 'onnx']
                        )
            
            assert result['creation_success'] is True
            assert 'torchscript' in result['model_exports']
            assert 'onnx' in result['model_exports']
            assert 'config.yaml' in result['package_contents']
            assert 'inference.py' in result['package_contents']
            assert 'requirements.txt' in result['package_contents']
            assert 'README.md' in result['package_contents']
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_create_inference_script(self, mock_text_processor, mock_model_factory):
        """Test inference script creation."""
        exporter = ModelExporter(self.config)
        script = exporter._create_inference_script()
        
        assert isinstance(script, str)
        assert "def synthesize_speech" in script
        assert "def load_torchscript_model" in script
        assert "def load_onnx_model" in script
        assert "if __name__ == \"__main__\":" in script
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_create_requirements(self, mock_text_processor, mock_model_factory):
        """Test requirements creation."""
        exporter = ModelExporter(self.config)
        requirements = exporter._create_requirements()
        
        assert isinstance(requirements, list)
        assert any("torch" in req for req in requirements)
        assert any("torchaudio" in req for req in requirements)
        assert any("numpy" in req for req in requirements)
    
    @patch('src.export.ModelFactory')
    @patch('src.export.TextProcessor')
    def test_create_deployment_readme(self, mock_text_processor, mock_model_factory):
        """Test deployment README creation."""
        exporter = ModelExporter(self.config)
        readme = exporter._create_deployment_readme()
        
        assert isinstance(readme, str)
        assert "# Speech Synthesis Deployment Package" in readme
        assert "## Contents" in readme
        assert "## Quick Start" in readme
        assert "## Model Formats" in readme


def test_run_torchscript_export():
    """Test run function for TorchScript export."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'system': {'device': 'cpu'}
    })
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = "output.pt"
    args.format = "torchscript"
    
    with patch('src.export.ModelExporter') as mock_exporter_class:
        mock_exporter = Mock()
        mock_exporter.export_torchscript.return_value = {
            'export_success': True,
            'file_size': 1024000
        }
        mock_exporter_class.return_value = mock_exporter
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_exporter.export_torchscript.assert_called_once_with("model.pt", "output.pt")


def test_run_onnx_export():
    """Test run function for ONNX export."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'system': {'device': 'cpu'}
    })
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = "output.onnx"
    args.format = "onnx"
    
    with patch('src.export.ModelExporter') as mock_exporter_class:
        mock_exporter = Mock()
        mock_exporter.export_onnx.return_value = {
            'export_success': True,
            'file_size': 2048000
        }
        mock_exporter_class.return_value = mock_exporter
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_exporter.export_onnx.assert_called_once_with("model.pt", "output.onnx")


def test_run_package_export():
    """Test run function for package export."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'system': {'device': 'cpu'}
    })
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = "package_dir"
    args.format = "package"
    args.include_formats = "torchscript,onnx"
    
    with patch('src.export.ModelExporter') as mock_exporter_class:
        mock_exporter = Mock()
        mock_exporter.create_deployment_package.return_value = {
            'creation_success': True,
            'model_exports': {'torchscript': {}, 'onnx': {}}
        }
        mock_exporter_class.return_value = mock_exporter
        
        result = run(args, config)
        
        assert result == 0  # Success
        mock_exporter.create_deployment_package.assert_called_once_with(
            "model.pt", "package_dir", ["torchscript", "onnx"]
        )


def test_run_missing_model_path():
    """Test run function with missing model path."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.model_path = None
    
    result = run(args, config)
    
    assert result == 1  # Error


def test_run_missing_output():
    """Test run function with missing output."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = None
    
    result = run(args, config)
    
    assert result == 1  # Error


def test_run_unknown_format():
    """Test run function with unknown export format."""
    config = OmegaConf.create({})
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = "output.xyz"
    args.format = "unknown_format"
    
    result = run(args, config)
    
    assert result == 1  # Error


def test_run_export_failure():
    """Test run function with export failure."""
    config = OmegaConf.create({
        'model': {'type': 'advanced'},
        'system': {'device': 'cpu'}
    })
    
    args = Mock()
    args.model_path = "model.pt"
    args.output = "output.pt"
    args.format = "torchscript"
    
    with patch('src.export.ModelExporter') as mock_exporter_class:
        mock_exporter = Mock()
        mock_exporter.export_torchscript.return_value = {
            'export_success': False,
            'error': 'Export failed'
        }
        mock_exporter_class.return_value = mock_exporter
        
        result = run(args, config)
        
        assert result == 1  # Error


if __name__ == "__main__":
    pytest.main([__file__])
