#!/usr/bin/env python3
"""
Advanced Speech Synthesis System - Main Entry Point

This is the main command-line interface for the Speech Synthesis project.
Provides comprehensive functionality for training, inference, voice cloning,
and web interface management.

Usage:
    python main.py [command] [options]

Commands:
    train       - Train a TTS model
    infer       - Synthesize speech from text
    clone       - Clone voice from reference audio
    web         - Launch web interface
    evaluate    - Evaluate model performance
    preprocess  - Preprocess dataset
    export      - Export model for production

Examples:
    python main.py web
    python main.py infer --text "Hello world" --output hello.wav
    python main.py train --config config.yaml
    python main.py clone --text "Test" --reference ref.wav --output cloned.wav
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src import train, infer, clone
    from src import evaluate, preprocess, export
    from src.gradio_interface import launch_interface
    from src.logging_config import get_logger
    from src.utils import setup_reproducibility, get_device
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global configuration
DEFAULT_CONFIG = "config.yaml"
LOGGER = None


def setup_global_config(config_path: Optional[str] = None, 
                       log_level: str = "INFO",
                       device: str = "auto",
                       seed: int = 42) -> dict:
    """Setup global configuration and logging."""
    global LOGGER
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        # Default configuration
        config = OmegaConf.create({
            'system': {
                'device': device,
                'log_level': log_level,
                'cache_dir': './cache',
                'max_workers': 4
            },
            'model': {
                'type': 'YourTTS',
                'checkpoint_path': None
            },
            'web': {
                'title': 'Speech Synthesis System',
                'description': 'Advanced TTS and Voice Cloning',
                'max_text_length': 1000,
                'enable_voice_cloning': True
            }
        })
    
    # Override with command line arguments
    if device != "auto":
        config.system.device = device
    if log_level != "INFO":
        config.system.log_level = log_level
    
    # Setup logging
    LOGGER = get_logger(config.get('system', {}))
    
    # Setup reproducibility
    setup_reproducibility(seed)
    
    LOGGER.logger.info(f"System initialized with device: {get_device(config.system.device)}")
    LOGGER.logger.info(f"Configuration loaded from: {config_path or 'defaults'}")
    
    return config


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments to parser."""
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')


def cmd_train(args):
    """Train a TTS model."""
    LOGGER.logger.info("Starting model training...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    # Override config with command line arguments
    if hasattr(args, 'output_dir') and args.output_dir:
        config.training.output_dir = args.output_dir
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    
    # Run training
    train.run(args, config)
    
    LOGGER.logger.info("Training completed successfully!")


def cmd_infer(args):
    """Synthesize speech from text."""
    LOGGER.logger.info("Starting speech synthesis...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    if not args.text:
        LOGGER.logger.error("Text input is required for synthesis")
        return 1
    
    if not args.output:
        args.output = "output.wav"
    
    # Run inference
    infer.run(args, config)
    
    LOGGER.logger.info(f"Speech synthesized successfully: {args.output}")


def cmd_clone(args):
    """Clone voice from reference audio."""
    LOGGER.logger.info("Starting voice cloning...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    if not args.text:
        LOGGER.logger.error("Text input is required for voice cloning")
        return 1
    
    if not args.reference:
        LOGGER.logger.error("Reference audio is required for voice cloning")
        return 1
    
    if not args.output:
        args.output = "cloned.wav"
    
    # Run voice cloning
    clone.run(args, config)
    
    LOGGER.logger.info(f"Voice cloned successfully: {args.output}")


def cmd_web(args):
    """Launch web interface."""
    LOGGER.logger.info("Launching web interface...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    # Launch interface
    try:
        launch_interface(
            config_path=args.config,
            share=getattr(args, 'share', False),
            server_name=getattr(args, 'host', '127.0.0.1'),
            server_port=getattr(args, 'port', 7860)
        )
    except KeyboardInterrupt:
        LOGGER.logger.info("Web interface stopped by user")
    except Exception as e:
        LOGGER.logger.error(f"Failed to launch web interface: {str(e)}")
        return 1


def cmd_evaluate(args):
    """Evaluate model performance."""
    LOGGER.logger.info("Starting model evaluation...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    # Run evaluation
    evaluate.run(args, config)
    
    LOGGER.logger.info("Evaluation completed successfully!")


def cmd_preprocess(args):
    """Preprocess dataset."""
    LOGGER.logger.info("Starting dataset preprocessing...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    # Run preprocessing
    preprocess.run(args, config)
    
    LOGGER.logger.info("Preprocessing completed successfully!")


def cmd_export(args):
    """Export model for production."""
    LOGGER.logger.info("Starting model export...")
    
    config = setup_global_config(args.config, args.log_level, args.device, args.seed)
    
    # Run export
    export.run(args, config)
    
    LOGGER.logger.info("Export completed successfully!")


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Speech Synthesis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s web                                    # Launch web interface
  %(prog)s infer --text "Hello" --output hi.wav  # Synthesize speech
  %(prog)s train --config config.yaml            # Train model
  %(prog)s clone --text "Test" --reference ref.wav --output clone.wav
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a TTS model")
    train.add_args(train_parser)
    add_common_args(train_parser)
    train_parser.add_argument('--output-dir', type=str, default='./outputs',
                             help='Output directory for model checkpoints')
    train_parser.add_argument('--batch-size', type=int,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float,
                             help='Learning rate for training')
    train_parser.add_argument('--epochs', type=int,
                             help='Number of training epochs')
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Synthesize speech from text")
    infer.add_args(infer_parser)
    add_common_args(infer_parser)
    infer_parser.add_argument('--speaker-id', type=int, default=0,
                             help='Speaker ID for multi-speaker models')
    infer_parser.add_argument('--speed', type=float, default=1.0,
                             help='Speech speed multiplier')
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone voice from reference audio")
    clone.add_args(clone_parser)
    add_common_args(clone_parser)
    clone_parser.add_argument('--similarity-threshold', type=float, default=0.7,
                             help='Minimum similarity threshold for cloning')
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    add_common_args(web_parser)
    web_parser.add_argument('--host', type=str, default='127.0.0.1',
                           help='Host address for web interface')
    web_parser.add_argument('--port', type=int, default=7860,
                           help='Port for web interface')
    web_parser.add_argument('--share', action='store_true',
                           help='Create public Gradio link')
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    add_common_args(eval_parser)
    eval_parser.add_argument('--checkpoint-path', type=str, 
                            help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', type=str,
                            help='Path to test dataset for evaluation')
    eval_parser.add_argument('--output-dir', type=str, default='evaluation_results',
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--benchmark', action='store_true',
                            help='Run benchmark evaluation instead of dataset evaluation')
    eval_parser.add_argument('--repetitions', type=int, default=10,
                            help='Number of repetitions for benchmark (default: 10)')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    add_common_args(preprocess_parser)
    preprocess_parser.add_argument('--input-dir', type=str, required=True,
                                  help='Input dataset directory')
    preprocess_parser.add_argument('--output-dir', type=str, required=True,
                                  help='Output preprocessed directory')
    preprocess_parser.add_argument('--format', type=str, default='auto',
                                  choices=['auto', 'ljspeech', 'common_voice', 'vctk', 'generic'],
                                  help='Dataset format (auto-detect by default)')
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model for production")
    add_common_args(export_parser)
    export_parser.add_argument('--model-path', type=str, required=True,
                              help='Path to model checkpoint')
    export_parser.add_argument('--format', type=str, 
                              choices=['torchscript', 'onnx', 'mobile', 'quantized', 'package'],
                              default='torchscript', help='Export format')
    export_parser.add_argument('--output', type=str, required=True,
                              help='Output file or directory path')
    export_parser.add_argument('--quantization-type', type=str, 
                              choices=['dynamic', 'static'], default='dynamic',
                              help='Quantization type (for quantized format)')
    export_parser.add_argument('--include-formats', type=str,
                              help='Comma-separated list of formats for package export')
    export_parser.add_argument('--optimize', action='store_true',
                              help='Apply optimizations for inference')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set verbose mode
    if hasattr(args, 'verbose') and args.verbose:
        args.log_level = 'DEBUG'
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0
    
    # Command dispatch
    commands = {
        'train': cmd_train,
        'infer': cmd_infer,
        'clone': cmd_clone,
        'web': cmd_web,
        'evaluate': cmd_evaluate,
        'preprocess': cmd_preprocess,
        'export': cmd_export
    }
    
    try:
        return commands[args.command](args) or 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        if LOGGER:
            LOGGER.logger.error(f"Unexpected error: {str(e)}")
        else:
            print(f"Error: {str(e)}")
        
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
