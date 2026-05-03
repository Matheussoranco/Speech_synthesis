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

from src.model import SynthesizerTrn
from src.text_processor import TextProcessor


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

        hp = config.model.get("params", {})
        model = SynthesizerTrn(
            n_vocab=hp.get("n_vocab", 512),
            spec_channels=hp.get("spec_channels", 513),
            segment_size=hp.get("segment_size", 8192),
            inter_channels=hp.get("inter_channels", 192),
            hidden_channels=hp.get("hidden_channels", 192),
            filter_channels=hp.get("filter_channels", 768),
            n_heads=hp.get("n_heads", 2),
            n_layers=hp.get("n_layers", 6),
            kernel_size=hp.get("kernel_size", 3),
            p_dropout=0.0,
            resblock=hp.get("resblock", "1"),
            resblock_kernel_sizes=tuple(hp.get("resblock_kernel_sizes", [3, 7, 11])),
            resblock_dilation_sizes=tuple(
                tuple(d) for d in hp.get("resblock_dilation_sizes", [[1,3,5],[1,3,5],[1,3,5]])
            ),
            upsample_rates=tuple(hp.get("upsample_rates", [8, 8, 2, 2])),
            upsample_initial_channel=hp.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=tuple(hp.get("upsample_kernel_sizes", [16, 16, 4, 4])),
            n_speakers=hp.get("n_speakers", 0),
            gin_channels=hp.get("gin_channels", 256),
            use_sdp=hp.get("use_sdp", True),
        )

        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("net_g", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)

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
            return np.zeros(1, dtype=np.float32)

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
    device = getattr(args, "device", "cpu")
    inferencer = TTSInferencer.from_checkpoint(args.model, device=device, config=config)
    out = inferencer.synthesize_to_file(
        args.text, args.output,
        sid=getattr(args, "speaker_id", None),
        noise_scale=getattr(args, "noise_scale", 0.667),
        noise_scale_w=getattr(args, "noise_scale_w", 0.8),
        length_scale=getattr(args, "length_scale", 1.0),
    )
    print(f"Saved: {out}")
