from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.vocoder.utils.io import load_config
import torch

class VocoderWrapper:
    """
    Wrapper for neural vocoders (e.g., HiFi-GAN).
    """
    def __init__(self, config_path, model_path, device='cpu'):
        self.config = load_config(config_path)
        self.model = setup_generator(model_path, self.config, device)
        self.model.eval()
        self.device = device

    def mel_to_audio(self, mel):
        with torch.no_grad():
            audio = self.model.inference(mel.to(self.device))
        return audio.cpu().numpy()
