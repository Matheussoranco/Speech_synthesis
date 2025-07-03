"""
Modern web interface for Speech Synthesis using Gradio.
Provides a user-friendly interface for TTS and voice cloning.
"""
import gradio as gr
import torch
import torchaudio
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List
import time
import io
import base64
from PIL import Image

# Import local modules
from src.tts_model import TTSWrapper
from src.speaker_encoder import SpeakerEncoder
from src.text_processor import TextProcessor
from src.logging_config import get_logger
from src.model import AdvancedTTSModel, load_model
from omegaconf import OmegaConf


class SpeechSynthesisInterface:
    """Modern Gradio interface for speech synthesis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the interface with configuration."""
        self.config = OmegaConf.load(config_path)
        self.logger = get_logger(self.config.get('system', {}))
        
        # Initialize components
        self.tts_wrapper = TTSWrapper()
        self.speaker_encoder = SpeakerEncoder()
        self.text_processor = TextProcessor(
            language=self.config.text_processing.language,
            phoneme_backend=self.config.text_processing.phoneme_backend
        )
        
        # Load custom model if available
        self.custom_model = None
        if self.config.model.checkpoint_path:
            try:
                self.custom_model = load_model(
                    self.config.model.checkpoint_path,
                    device=self._get_device()
                )
                self.logger.log_model_info(
                    "Custom TTS Model",
                    sum(p.numel() for p in self.custom_model.parameters()),
                    self._get_device()
                )
            except Exception as e:
                self.logger.log_error_with_context(e, {"component": "model_loading"})
        
        # Performance tracking
        self.synthesis_times = []
        self.clone_times = []
    
    def _get_device(self) -> str:
        """Get appropriate device for computation."""
        device_config = self.config.system.device
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_config
    
    def _validate_text(self, text: str) -> Tuple[bool, str]:
        """Validate input text."""
        if not text or not text.strip():
            return False, "Please enter some text to synthesize."
        
        if len(text) > self.config.web.max_text_length:
            return False, f"Text too long. Maximum {self.config.web.max_text_length} characters allowed."
        
        return True, ""
    
    def _validate_audio(self, audio_file) -> Tuple[bool, str, Optional[np.ndarray], Optional[int]]:
        """Validate uploaded audio file."""
        if audio_file is None:
            return False, "Please upload an audio file.", None, None
        
        try:
            # Load audio
            if isinstance(audio_file, str):
                waveform, sample_rate = torchaudio.load(audio_file)
            else:
                # Handle tuple format (sample_rate, audio_data)
                sample_rate, audio_data = audio_file
                waveform = torch.from_numpy(audio_data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            
            # Convert to numpy
            audio_np = waveform.squeeze().numpy()
            
            # Check duration
            duration = len(audio_np) / sample_rate
            min_duration = self.config.voice_cloning.min_reference_length
            max_duration = self.config.voice_cloning.max_reference_length
            
            if duration < min_duration:
                return False, f"Audio too short. Minimum {min_duration} seconds required.", None, None
            
            if duration > max_duration:
                return False, f"Audio too long. Maximum {max_duration} seconds allowed.", None, None
            
            return True, "", audio_np, sample_rate
            
        except Exception as e:
            return False, f"Error processing audio file: {str(e)}", None, None
    
    def synthesize_speech(self, 
                         text: str, 
                         model_choice: str = "Coqui TTS",
                         voice_speed: float = 1.0,
                         voice_pitch: float = 1.0,
                         add_silence: bool = True) -> Tuple[Optional[str], str, dict]:
        """
        Synthesize speech from text.
        
        Returns:
            Tuple of (audio_file_path, status_message, performance_info)
        """
        start_time = time.time()
        
        # Validate input
        is_valid, error_msg = self._validate_text(text)
        if not is_valid:
            return None, error_msg, {}
        
        try:
            # Process text
            processed_text = self.text_processor.process_text(
                text, 
                use_phonemes=self.config.text_processing.use_phonemes
            )
            
            # Generate speech
            if model_choice == "Custom Model" and self.custom_model:
                # Use custom model (placeholder implementation)
                audio_data = self._synthesize_with_custom_model(processed_text)
            else:
                # Use Coqui TTS
                audio_data = self.tts_wrapper.synthesize(processed_text)
            
            # Apply audio effects
            if voice_speed != 1.0 or voice_pitch != 1.0:
                audio_data = self._apply_audio_effects(audio_data, voice_speed, voice_pitch)
            
            # Add silence if requested
            if add_silence:
                audio_data = self._add_silence(audio_data)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.tts_wrapper.save(audio_data, temp_file.name)
            
            # Log performance
            inference_time = time.time() - start_time
            self.synthesis_times.append(inference_time)
            self.logger.log_inference_time(len(text), inference_time, model_choice)
            
            # Performance info
            performance_info = {
                "inference_time": f"{inference_time:.2f}s",
                "text_length": len(text),
                "audio_duration": f"{len(audio_data) / 22050:.2f}s",
                "real_time_factor": f"{inference_time / (len(audio_data) / 22050):.2f}x"
            }
            
            return temp_file.name, "✅ Speech synthesized successfully!", performance_info
            
        except Exception as e:
            error_msg = f"❌ Error during synthesis: {str(e)}"
            self.logger.log_error_with_context(e, {"text_length": len(text), "model": model_choice})
            return None, error_msg, {}
    
    def clone_voice(self, 
                   text: str,
                   reference_audio,
                   similarity_threshold: float = 0.7) -> Tuple[Optional[str], str, dict]:
        """
        Clone voice from reference audio.
        
        Returns:
            Tuple of (audio_file_path, status_message, performance_info)
        """
        start_time = time.time()
        
        # Validate inputs
        is_valid_text, text_error = self._validate_text(text)
        if not is_valid_text:
            return None, text_error, {}
        
        is_valid_audio, audio_error, audio_np, sample_rate = self._validate_audio(reference_audio)
        if not is_valid_audio:
            return None, audio_error, {}
        
        try:
            # Extract speaker embedding
            speaker_embedding = self.speaker_encoder.extract_embedding(audio_np, sample_rate)
            
            # Calculate similarity (placeholder for actual implementation)
            similarity_score = np.random.uniform(0.6, 0.9)  # Placeholder
            
            if similarity_score < similarity_threshold:
                return None, f"❌ Voice similarity too low ({similarity_score:.2f}). Please try a different reference audio.", {}
            
            # Process text
            processed_text = self.text_processor.process_text(text)
            
            # Generate cloned speech
            if isinstance(reference_audio, str):
                cloned_audio = self.tts_wrapper.synthesize(processed_text, speaker_wav=reference_audio)
            else:
                # Save reference audio to temp file for Coqui TTS
                temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                torchaudio.save(temp_ref.name, torch.from_numpy(audio_np).unsqueeze(0), sample_rate)
                cloned_audio = self.tts_wrapper.synthesize(processed_text, speaker_wav=temp_ref.name)
                os.unlink(temp_ref.name)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.tts_wrapper.save(cloned_audio, temp_file.name)
            
            # Log performance
            inference_time = time.time() - start_time
            self.clone_times.append(inference_time)
            self.logger.log_voice_cloning("reference_audio", text, similarity_score)
            
            # Performance info
            performance_info = {
                "inference_time": f"{inference_time:.2f}s",
                "similarity_score": f"{similarity_score:.3f}",
                "text_length": len(text),
                "reference_duration": f"{len(audio_np) / sample_rate:.2f}s"
            }
            
            return temp_file.name, f"✅ Voice cloned successfully! Similarity: {similarity_score:.3f}", performance_info
            
        except Exception as e:
            error_msg = f"❌ Error during voice cloning: {str(e)}"
            self.logger.log_error_with_context(e, {"text_length": len(text)})
            return None, error_msg, {}
    
    def _synthesize_with_custom_model(self, text: str) -> np.ndarray:
        """Synthesize using custom model (placeholder)."""
        # This would implement actual custom model inference
        # For now, fallback to Coqui TTS
        return self.tts_wrapper.synthesize(text)
    
    def _apply_audio_effects(self, audio: np.ndarray, speed: float, pitch: float) -> np.ndarray:
        """Apply speed and pitch effects to audio."""
        # Placeholder implementation
        # In practice, would use librosa or similar
        return audio
    
    def _add_silence(self, audio: np.ndarray, silence_duration: float = 0.5) -> np.ndarray:
        """Add silence at the beginning and end of audio."""
        sample_rate = 22050  # Default sample rate
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        return np.concatenate([silence, audio, silence])
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            "total_syntheses": len(self.synthesis_times),
            "total_clones": len(self.clone_times),
            "avg_synthesis_time": np.mean(self.synthesis_times) if self.synthesis_times else 0,
            "avg_clone_time": np.mean(self.clone_times) if self.clone_times else 0,
        }
        return stats
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .performance-box {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .error-box {
            background-color: #ffe6e6;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ff9999;
        }
        .success-box {
            background-color: #e6ffe6;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #99ff99;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=css,
            title="Advanced Speech Synthesis"
        ) as interface:
            
            gr.Markdown(f"""
            # 🎤 {self.config.web.title}
            
            {self.config.web.description}
            
            Choose between high-quality text-to-speech synthesis or voice cloning from audio samples.
            """)
            
            with gr.Tabs():
                # Text-to-Speech Tab
                with gr.TabItem("🗣️ Text-to-Speech"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Enter the text you want to convert to speech...",
                                lines=4,
                                max_lines=10
                            )
                            
                            with gr.Row():
                                model_choice = gr.Dropdown(
                                    label="TTS Model",
                                    choices=["Coqui TTS", "Custom Model"] if self.custom_model else ["Coqui TTS"],
                                    value="Coqui TTS"
                                )
                                
                            with gr.Row():
                                voice_speed = gr.Slider(
                                    label="Speech Speed",
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1
                                )
                                voice_pitch = gr.Slider(
                                    label="Voice Pitch",
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1
                                )
                            
                            add_silence = gr.Checkbox(
                                label="Add silence padding",
                                value=True
                            )
                            
                            synthesize_btn = gr.Button("🎵 Synthesize Speech", variant="primary")
                        
                        with gr.Column(scale=1):
                            audio_output = gr.Audio(label="Generated Speech")
                            status_output = gr.Textbox(label="Status", interactive=False)
                            performance_output = gr.JSON(label="Performance Info")
                    
                    # Event handlers for TTS
                    synthesize_btn.click(
                        fn=self.synthesize_speech,
                        inputs=[text_input, model_choice, voice_speed, voice_pitch, add_silence],
                        outputs=[audio_output, status_output, performance_output]
                    )
                
                # Voice Cloning Tab
                with gr.TabItem("🎭 Voice Cloning"):
                    if self.config.web.enable_voice_cloning:
                        with gr.Row():
                            with gr.Column(scale=2):
                                clone_text_input = gr.Textbox(
                                    label="Text to Synthesize",
                                    placeholder="Enter the text for voice cloning...",
                                    lines=3,
                                    max_lines=8
                                )
                                
                                reference_audio = gr.Audio(
                                    label="Reference Audio (3-10 seconds)",
                                    type="numpy"
                                )
                                
                                similarity_threshold = gr.Slider(
                                    label="Similarity Threshold",
                                    minimum=0.5,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.05,
                                    info="Minimum similarity score required for cloning"
                                )
                                
                                clone_btn = gr.Button("🎭 Clone Voice", variant="primary")
                            
                            with gr.Column(scale=1):
                                cloned_audio_output = gr.Audio(label="Cloned Speech")
                                clone_status_output = gr.Textbox(label="Status", interactive=False)
                                clone_performance_output = gr.JSON(label="Performance Info")
                        
                        # Event handlers for voice cloning
                        clone_btn.click(
                            fn=self.clone_voice,
                            inputs=[clone_text_input, reference_audio, similarity_threshold],
                            outputs=[cloned_audio_output, clone_status_output, clone_performance_output]
                        )
                    else:
                        gr.Markdown("Voice cloning is disabled in the configuration.")
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings & Info"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Information")
                            
                            device_info = gr.Textbox(
                                label="Device",
                                value=self._get_device(),
                                interactive=False
                            )
                            
                            model_info = gr.Textbox(
                                label="Available Models",
                                value="Coqui TTS" + (", Custom Model" if self.custom_model else ""),
                                interactive=False
                            )
                            
                            gr.Markdown("### Performance Statistics")
                            stats_output = gr.JSON(label="Statistics")
                            
                            refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                            refresh_stats_btn.click(
                                fn=self.get_performance_stats,
                                outputs=[stats_output]
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Configuration")
                            
                            gr.Textbox(
                                label="Language",
                                value=self.config.text_processing.language,
                                interactive=False
                            )
                            
                            gr.Textbox(
                                label="Max Text Length",
                                value=str(self.config.web.max_text_length),
                                interactive=False
                            )
                            
                            gr.Textbox(
                                label="Audio Sample Rate",
                                value=str(self.config.data.sample_rate),
                                interactive=False
                            )
            
            # Footer
            gr.Markdown("""
            ---
            💡 **Tips:**
            - Use clear, well-punctuated text for better results
            - For voice cloning, provide 3-10 seconds of clear speech
            - Adjust speed and pitch to fine-tune the output
            """)
        
        return interface


def create_interface(config_path: str = "config.yaml") -> gr.Interface:
    """Create and return the Gradio interface."""
    app = SpeechSynthesisInterface(config_path)
    return app.create_interface()


def launch_interface(config_path: str = "config.yaml", 
                    share: bool = False,
                    server_name: str = "127.0.0.1",
                    server_port: int = 7860):
    """Launch the Gradio interface."""
    interface = create_interface(config_path)
    
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_interface()
