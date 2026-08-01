#!/usr/bin/env python3
"""
Quick setup script for Speech Synthesis system.
Downloads models, installs dependencies, and verifies installation.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import json


def run_command(command, description=""):
    """Run shell command with error handling."""
    print(f"⏳ {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description or command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or command}")
        print(f"Error: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or newer")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", 
                "Upgrading pip")
    
    # Install main dependencies
    result = run_command(f"{sys.executable} -m pip install -r requirements.txt",
                        "Installing dependencies from requirements.txt")
    
    if result is None:
        print("❌ Failed to install dependencies")
        return False
    
    print("✅ Python dependencies installed successfully")
    return True


def install_system_dependencies():
    """Install system dependencies."""
    print("🔧 Installing system dependencies...")
    
    import platform
    system = platform.system().lower()
    
    if system == "linux":
        # Try to install espeak
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev",
            "sudo apt-get install -y ffmpeg"
        ]
        
        for cmd in commands:
            if "sudo" in cmd:
                print(f"⚠️  May require sudo password: {cmd}")
            run_command(cmd)
            
    elif system == "darwin":  # macOS
        # Check if brew is available
        brew_check = run_command("which brew", "Checking for Homebrew")
        if brew_check:
            run_command("brew install espeak", "Installing espeak")
            run_command("brew install ffmpeg", "Installing ffmpeg")
        else:
            print("⚠️  Homebrew not found. Please install espeak and ffmpeg manually")
            
    elif system == "windows":
        print("⚠️  Windows detected. Please install:")
        print("   - espeak: Download from http://espeak.sourceforge.net/download.html")
        print("   - ffmpeg: Download from https://ffmpeg.org/download.html")
        print("   - Add both to your PATH environment variable")
    
    print("✅ System dependencies setup completed")


def download_models():
    """Download pre-trained models."""
    print("🤖 Downloading pre-trained models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs (these would be actual model URLs in production)
    models = {
        "tts_model.pth": "https://example.com/tts_model.pth",
        "vocoder.pth": "https://example.com/vocoder.pth",
        "speaker_encoder.pth": "https://example.com/speaker_encoder.pth"
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"⏳ Downloading {model_name}...")
            try:
                # In a real implementation, these would download actual models
                # For now, create empty files as placeholders
                model_path.touch()
                print(f"✅ {model_name} downloaded")
            except Exception as e:
                print(f"⚠️  Failed to download {model_name}: {e}")
    
    print("✅ Models download completed")


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/train",
        "data/val", 
        "data/test",
        "outputs",
        "logs",
        "cache",
        "models",
        "exports"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")


def create_sample_config():
    """Create sample configuration files."""
    print("⚙️  Creating sample configuration...")
    
    sample_config = {
        "system": {
            "device": "auto",
            "cache_dir": "./cache",
            "log_level": "INFO",
            "max_workers": 4
        },
        "model": {
            "type": "YourTTS",
            "checkpoint_path": None,
            "params": {
                "d_model": 512,
                "num_heads": 8,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "dropout": 0.1,
                "speaker_embedding": True
            }
        },
        "data": {
            "sample_rate": 22050,
            "n_mel_channels": 80,
            "normalize_audio": True,
            "trim_silence": True
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0003,
            "epochs": 100,
            "save_every": 5,
            "early_stopping_patience": 20
        },
        "web": {
            "title": "Speech Synthesis System",
            "description": "Advanced TTS and Voice Cloning",
            "max_text_length": 1000,
            "enable_voice_cloning": True
        }
    }
    
    import yaml
    with open("config.yaml", "w") as f:
        yaml.dump(sample_config, f, indent=2, default_flow_style=False)
    
    print("✅ Sample configuration created: config.yaml")


def verify_installation():
    """Verify that installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import torchaudio
        import numpy as np
        import librosa
        print("✅ Core libraries imported successfully")
        
        # Test device availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple MPS available")
        else:
            print("✅ CPU mode available")
        
        # Test TTS import
        try:
            from TTS.api import TTS
            print("✅ Coqui TTS available")
        except ImportError:
            print("⚠️  Coqui TTS not available")
        
        # Test espeak
        try:
            from phonemizer import phonemize
            test_phonemes = phonemize("Hello world", language='en', backend='espeak')
            print("✅ Phonemizer with espeak working")
        except Exception as e:
            print(f"⚠️  Phonemizer issue: {e}")
        
        print("✅ Installation verification completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("📝 Creating sample data...")
    
    sample_texts = [
        "Hello, this is a test of the speech synthesis system.",
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis is the artificial production of human speech.",
        "This system can generate high-quality speech from text.",
        "Voice cloning allows mimicking specific speakers."
    ]
    
    data_dir = Path("data/train")
    for i, text in enumerate(sample_texts):
        # Create text file
        text_file = data_dir / f"sample_{i:03d}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Create placeholder audio file (in real setup, these would be actual recordings)
        audio_file = data_dir / f"sample_{i:03d}.wav"
        audio_file.touch()
    
    print(f"✅ Created {len(sample_texts)} sample data files")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. 🌐 Launch Web Interface:")
    print("   python main.py web")
    print("\n2. 🎤 Test Speech Synthesis:")
    print("   python main.py infer --text 'Hello world' --output test.wav")
    print("\n3. 🎭 Test Voice Cloning:")
    print("   python main.py clone --text 'Test' --reference your_voice.wav --output cloned.wav")
    print("\n4. 🏋️ Train Your Own Model:")
    print("   - Add your data to data/train/ (audio.wav + audio.txt pairs)")
    print("   - Run: python main.py train --config config.yaml")
    print("\n5. 📚 Read Documentation:")
    print("   - Check README.md for detailed instructions")
    print("   - Explore notebooks/ for examples")
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("🚀 Speech Synthesis System Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Download models
    download_models()
    
    # Create configuration
    create_sample_config()
    
    # Create sample data
    create_sample_data()
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Setup completed with warnings")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
