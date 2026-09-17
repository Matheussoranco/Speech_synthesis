"""
VITS2 inference: text → high-fidelity waveform.

For a given text string:
  1. G2P: text → phoneme IDs
  2. VITS2.infer(x, g): prior sampling + flow inversion + HiFi-GAN decoding
  3. Write WAV at 22050 Hz

Noise parameters
  noise_scale   σ for the latent prior (0 = deterministic, 1 = high variation)
  noise_scale_w σ for the duration sampler (Stochastic Duration Predictor)
  length_scale  multiply phoneme durations  (>1 slower, <1 faster)
"""
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf

from .model import SynthesizerTrn
from .text_processor import TextProcessor
from .utils import secure_torch_load


class TTSInferencer:
    """Wraps a trained VITS2 model for text-to-speech synthesis."""

    def __init__(
        self,
        model: SynthesizerTrn,
        text_processor: TextProcessor,
        sample_rate: int = 22050,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.tp = text_processor
        self.sample_rate = sample_rate
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        config=None,
    ) -> "TTSInferencer":
        if config is None:
            from omegaconf import OmegaConf
            config = OmegaConf.load(
                Path(checkpoint_path).parent.parent / "config.yaml"
            )

        from .from_checkpoint import build_synthesizer, load_synthesizer_state

        model = build_synthesizer(config, p_dropout=0.0)
        load_synthesizer_state(model, checkpoint_path, device=device, strict=False)

        tp = TextProcessor(
            language=config.text_processing.get("language", "en"),
            phoneme_backend=config.text_processing.get("phoneme_backend", "espeak"),
        )
        return cls(model, tp, config.data.get("sample_rate", 22050), device)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        sid: Optional[int] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Returns float32 np.ndarray waveform at self.sample_rate.
        """
        phonemes = self.tp.text_to_phonemes(text)
        ids = self.tp.phonemes_to_ids(phonemes)
        if not ids:
            raise ValueError(
                "no phoneme ids produced for the input text — refusing to "
                "return a silent zeros(1) placeholder; check language/backend "
                f"and input text {text!r:.80}"
            )

        x = torch.LongTensor(ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(ids)]).to(self.device)

        sid_t = None
        if sid is not None and self.model.n_speakers > 1:
            sid_t = torch.LongTensor([sid]).to(self.device)

        wav, _, _ = self.model.infer(
            x, x_lengths, sid=sid_t,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        return wav.squeeze().cpu().float().numpy()

    def synthesize_to_file(self, text: str, output_path: str, **kwargs) -> str:
        wav = self.synthesize(text, **kwargs)
        sf.write(output_path, wav, self.sample_rate)
        return output_path


# ---------------------------------------------------------------------------
# CLI shim
# ---------------------------------------------------------------------------

def add_args(parser):
    parser.add_argument("--model", type=str, required=True,
                        help="Path to checkpoint (outputs/best.pth)")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--speaker-id", type=int, default=None)


def run(args, config=None):
    # args.device is the raw CLI string (default "auto"); torch.device("auto")
    # raises, and every downstream .to(device) call needs a real device, so
    # resolve it the same way train.py does before handing it off.
    from .utils import get_device
    device = str(get_device(getattr(args, "device", "cpu")))
    inferencer = TTSInferencer.from_checkpoint(args.model, device=device, config=config)
    out = inferencer.synthesize_to_file(
        args.text, args.output,
        sid=getattr(args, "speaker_id", None),
        noise_scale=getattr(args, "noise_scale", 0.667),
        noise_scale_w=getattr(args, "noise_scale_w", 0.8),
        length_scale=getattr(args, "length_scale", 1.0),
    )
    print(f"Saved: {out}")
