#!/usr/bin/env python3
"""
Example script demonstrating Speech Synthesis System usage.
Shows various use cases and best practices.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def example_basic_synthesis():
    """Example 1: Basic text-to-speech synthesis."""
    print("🎤 Example 1: Basic Text-to-Speech")
    print("-" * 40)
    
    try:
        from src.tts_model import TTSWrapper
        
        # Initialize TTS
        tts = TTSWrapper()
        
        # Simple synthesis
        text = "Hello! This is a demonstration of our speech synthesis system."
        print(f"Synthesizing: '{text}'")
        
        audio = tts.synthesize(text)
        output_path = "examples/basic_synthesis.wav"
        
        # Create output directory
        Path("examples").mkdir(exist_ok=True)
        
        tts.save(audio, output_path)
        print(f"✅ Audio saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_voice_cloning():
    """Example 2: Voice cloning demonstration."""
    print("🎭 Example 2: Voice Cloning")
    print("-" * 40)
    
    try:
        from src.tts_model import TTSWrapper
        from src.speaker_encoder import SpeakerEncoder
        
        # Initialize components
        tts = TTSWrapper()
        speaker_encoder = SpeakerEncoder()
        
        # Text to synthesize
        text = "This is speech with a cloned voice."
        print(f"Text: '{text}'")
        
        # For this example, we'll create a dummy reference
        # In practice, you'd use a real audio file
        reference_path = "examples/reference_dummy.wav"
        
        print("ℹ️  Note: This example uses a dummy reference.")
        print("   In practice, provide a real audio file (3-10 seconds).")
        
        # Simulate voice cloning
        # cloned_audio = tts.synthesize(text, speaker_wav=reference_path)
        # output_path = "examples/voice_cloned.wav"
        # tts.save(cloned_audio, output_path)
        
        print("✅ Voice cloning example prepared")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_text_processing():
    """Example 3: Advanced text processing."""
    print("📝 Example 3: Text Processing")
    print("-" * 40)
    
    try:
        from src.text_processor import TextProcessor
        
        # Initialize text processor
        processor = TextProcessor(
            language="en",
            normalize_numbers=True,
            normalize_abbreviations=True
        )
        
        # Test text with various challenges
        test_texts = [
            "Dr. Smith bought 2 apples for $3.50 on 12/25/2023.",
            "The meeting is at 3:30 PM in room no. 42.",
            "I have 1,500 items that cost approx. $25.99 each.",
            "Visit www.example.com or call +1-555-123-4567.",
        ]
        
        print("Original → Processed:")
        for text in test_texts:
            processed = processor.process_text(text)
            print(f"'{text}'")
            print(f"→ '{processed}'")
            print()
        
        # Text statistics
        sample_text = "This is a sample text for analysis."
        stats = processor.get_text_statistics(sample_text)
        print("Text Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("✅ Text processing example completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_batch_processing():
    """Example 4: Batch processing multiple texts."""
    print("📦 Example 4: Batch Processing")
    print("-" * 40)
    
    try:
        from src.tts_model import TTSWrapper
        
        # Initialize TTS
        tts = TTSWrapper()
        
        # Multiple texts to process
        texts = [
            "First sentence for batch processing.",
            "Second sentence with different content.",
            "Third and final sentence in the batch.",
        ]
        
        print(f"Processing {len(texts)} texts...")
        
        # Create output directory
        output_dir = Path("examples/batch_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for i, text in enumerate(texts):
            print(f"  Processing {i+1}/{len(texts)}: '{text[:30]}...'")
            
            audio = tts.synthesize(text)
            output_path = output_dir / f"batch_{i:03d}.wav"
            tts.save(audio, str(output_path))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Batch processing completed in {total_time:.2f} seconds")
        print(f"   Average: {total_time/len(texts):.2f} seconds per text")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_multilingual():
    """Example 5: Multi-language synthesis."""
    print("🌍 Example 5: Multi-language Support")
    print("-" * 40)
    
    try:
        from src.text_processor import TextProcessor
        from src.tts_model import TTSWrapper
        
        # Different languages
        languages = {
            "en": "Hello, how are you today?",
            "es": "Hola, ¿cómo estás hoy?",
            "fr": "Bonjour, comment allez-vous aujourd'hui?",
            "de": "Hallo, wie geht es dir heute?",
        }
        
        # Initialize TTS
        tts = TTSWrapper()
        
        for lang_code, text in languages.items():
            print(f"Language: {lang_code.upper()}")
            print(f"Text: '{text}'")
            
            # Process text for language
            processor = TextProcessor(language=lang_code)
            processed_text = processor.process_text(text)
            print(f"Processed: '{processed_text}'")
            
            # Note: In practice, you'd need language-specific models
            print("ℹ️  Language-specific synthesis would happen here")
            print()
        
        print("✅ Multi-language example completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_performance_monitoring():
    """Example 6: Performance monitoring."""
    print("📊 Example 6: Performance Monitoring")
    print("-" * 40)
    
    try:
        from src.tts_model import TTSWrapper
        import time
        
        # Initialize TTS
        tts = TTSWrapper()
        
        # Test different text lengths
        test_cases = [
            "Short text.",
            "This is a medium length text for testing synthesis performance.",
            "This is a much longer text that contains multiple sentences and should take more time to process. It includes various words and punctuation marks to test the complete synthesis pipeline.",
        ]
        
        print("Performance Analysis:")
        print("Text Length | Synthesis Time | Real-time Factor")
        print("-" * 50)
        
        for text in test_cases:
            start_time = time.time()
            audio = tts.synthesize(text)
            synthesis_time = time.time() - start_time
            
            # Estimate audio duration (approximate)
            estimated_duration = len(text) * 0.05  # ~50ms per character
            rtf = synthesis_time / estimated_duration if estimated_duration > 0 else 0
            
            print(f"{len(text):11d} | {synthesis_time:13.3f}s | {rtf:15.2f}x")
        
        print("✅ Performance monitoring completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_configuration():
    """Example 7: Configuration management."""
    print("⚙️ Example 7: Configuration Management")
    print("-" * 40)
    
    try:
        from omegaconf import OmegaConf
        
        # Load configuration
        if os.path.exists("config.yaml"):
            config = OmegaConf.load("config.yaml")
            print("✅ Configuration loaded from config.yaml")
        else:
            # Create sample configuration
            config = OmegaConf.create({
                "system": {
                    "device": "auto",
                    "log_level": "INFO"
                },
                "model": {
                    "type": "YourTTS",
                    "params": {
                        "d_model": 512,
                        "num_heads": 8
                    }
                },
                "training": {
                    "batch_size": 16,
                    "learning_rate": 0.0003
                }
            })
            print("📝 Sample configuration created")
        
        # Display configuration
        print("\nCurrent Configuration:")
        print(OmegaConf.to_yaml(config))
        
        # Modify configuration programmatically
        config.training.batch_size = 32
        config.system.device = "cpu"
        
        print("Modified Configuration:")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Device: {config.system.device}")
        
        print("✅ Configuration management example completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("🚀 Speech Synthesis System Examples")
    print("=" * 50)
    print()
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    # Run examples
    examples = [
        example_basic_synthesis,
        example_voice_cloning,
        example_text_processing,
        example_batch_processing,
        example_multilingual,
        example_performance_monitoring,
        example_configuration,
    ]
    
    for i, example_func in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] ", end="")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print()
    
    print("🎉 All examples completed!")
    print("\n💡 Tips:")
    print("  - Check the 'examples/' directory for generated files")
    print("  - Modify these examples for your specific use case")
    print("  - Refer to the documentation for advanced features")
    print()


if __name__ == "__main__":
    main()
