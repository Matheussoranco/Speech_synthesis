"""
Zero-shot voice cloning via ECAPA-TDNN speaker embeddings.

Pipeline:
  1. Load reference audio  →  resample to 16 kHz
  2. ECAPA-TDNN encoder   →  192-dim d-vector  g ∈ ℝ¹⁹²
  3. Text → phoneme tokens  x ∈ ℤᵀˣ
  4. VITS2 infer(x, g)     →  waveform ŷ
  5. Save ŷ as WAV

The speaker embedding g conditions both:
  - Prior encoder p_θ(z | c_text, g)   via additive bias in WN layers
  - Decoder p_θ(x | z, g)              via conditional convolution

Because VITS2 is trained with g from the ground-truth speaker, zero-shot
cloning works by substituting the reference embedding at inference time.
Similarity between the cloned voice and reference is measured via cosine
distance in embedding space.
"""
import os
import math
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np
import soundfile as sf

from src.model import SynthesizerTrn
from src.speaker_encoder import SpeakerEncoder
from src.text_processor import TextProcessor


class VoiceCloner:
    """
    High-level voice cloning interface.

    Usage:
        cloner = VoiceCloner.from_checkpoint("outputs/best.pth")
        wav = cloner.clone("Hello, world!", reference_path="ref.wav")
        sf.write("out.wav", wav, 22050)
    """

    def __init__(
        self,
        synthesizer: SynthesizerTrn,
        speaker_encoder: SpeakerEncoder,
        text_processor: TextProcessor,
        sample_rate: int = 22050,
        device: str = "cpu",
    ):
        self.syn = synthesizer.to(device).eval()
        self.spk_enc = speaker_encoder.to(device).eval()
        self.tp = text_processor
        self.sample_rate = sample_rate
        self.device = device

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        speaker_encoder_path: Optional[str] = None,
        device: str = "cpu",
        config=None,
    ) -> "VoiceCloner":
        """Load cloner from a VITS2 checkpoint."""
        if config is None:
            from omegaconf import OmegaConf
            config = OmegaConf.load(
                Path(checkpoint_path).parent.parent / "config.yaml"
            )

        from src.model import SynthesizerTrn
        hp = config.model.get("params", {})
        syn = SynthesizerTrn(
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
        syn.load_state_dict(ckpt["net_g"])

        spk_enc = SpeakerEncoder(
            sample_rate=16000,
            n_mels=80,
            channels=1024,
            emb_dim=192,
            device=device,
        )
        if speaker_encoder_path and os.path.exists(speaker_encoder_path):
            spk_enc.model.load_state_dict(
                torch.load(speaker_encoder_path, map_location=device)
            )

        tp = TextProcessor(
            language=config.text_processing.get("language", "en"),
            phoneme_backend=config.text_processing.get("phoneme_backend", "espeak"),
        )

        return cls(syn, spk_enc, tp, config.data.get("sample_rate", 22050), device)

    # ------------------------------------------------------------------
    def _load_reference(self, path: str) -> torch.Tensor:
        """Load reference audio, resample to 16 kHz mono."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0)

    def embed_reference(self, reference_path: str) -> torch.Tensor:
        """Extract 192-dim d-vector from reference audio file."""
        wav = self._load_reference(reference_path).to(self.device)
        with torch.no_grad():
            emb = self.spk_enc.embed_utterance(wav)  # (192,)
        return emb

    # ------------------------------------------------------------------
    @torch.no_grad()
    def clone(
        self,
        text: str,
        reference_path: str,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        similarity_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Clone voice from reference audio.

        Parameters
        ----------
        text:               Input text to synthesize.
        reference_path:     Path to reference WAV (≥ 3 s recommended).
        noise_scale:        σ for prior sampling (lower = less varied).
        noise_scale_w:      σ for duration sampling.
        length_scale:       > 1 slows speech, < 1 speeds it up.
        similarity_threshold: minimum cosine-sim check (0 = disabled).

        Returns
        -------
        np.ndarray  float32 waveform at self.sample_rate.
        """
        # 1. Speaker embedding
        g = self.embed_reference(reference_path).unsqueeze(0)  # (1, 192)

        # 2. Phoneme sequence
        phonemes = self.tp.text_to_phonemes(text)
        ids = self.tp.phonemes_to_ids(phonemes)
        x = torch.LongTensor(ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(ids)]).to(self.device)

        # 3. Synthesize
        wav, _, _ = self.syn.infer(
            x, x_lengths, g=g,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        wav_np = wav.squeeze().cpu().float().numpy()

        # 4. Optional similarity check
        if similarity_threshold > 0:
            sim = self._speaker_similarity(wav_np, reference_path)
            if sim < similarity_threshold:
                import warnings
                warnings.warn(
                    f"Speaker similarity {sim:.3f} below threshold "
                    f"{similarity_threshold:.3f}. Consider more reference audio."
                )
        return wav_np

    def _speaker_similarity(self, synthesized: np.ndarray, reference_path: str) -> float:
        """Cosine similarity between synthesized and reference d-vectors."""
        import torchaudio.functional as AF
        ref_wav = self._load_reference(reference_path).to(self.device)
        syn_t = torch.from_numpy(synthesized).to(self.device)
        # Resample synthesized (at self.sample_rate) to 16 kHz
        if self.sample_rate != 16000:
            syn_t = AF.resample(syn_t, self.sample_rate, 16000)

        e_ref = self.embed_reference(reference_path)
        e_syn = self.spk_enc.embed_utterance(syn_t)
        return torch.nn.functional.cosine_similarity(e_ref.unsqueeze(0),
                                                      e_syn.unsqueeze(0)).item()

    def clone_to_file(
        self,
        text: str,
        reference_path: str,
        output_path: str,
        **kwargs,
    ) -> float:
        """Clone and write WAV. Returns speaker similarity score."""
        wav = self.clone(text, reference_path, **kwargs)
        sf.write(output_path, wav, self.sample_rate)
        sim = self._speaker_similarity(wav, reference_path)
        return sim


# ---------------------------------------------------------------------------
# CLI shim
# ---------------------------------------------------------------------------

def add_args(parser):
    parser.add_argument("--model", type=str, required=True,
                        help="Path to VITS2 checkpoint (outputs/best.pth)")
    parser.add_argument("--reference", type=str, required=True,
                        help="Reference WAV for voice cloning")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize in the cloned voice")
    parser.add_argument("--output", type=str, default="cloned.wav",
                        help="Output WAV path")
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.0)


def run(args, config=None):
    device = getattr(args, "device", "cpu")
    cloner = VoiceCloner.from_checkpoint(args.model, device=device, config=config)
    sim = cloner.clone_to_file(
        args.text, args.reference, args.output,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        similarity_threshold=args.similarity_threshold,
    )
    print(f"Saved {args.output}  (speaker similarity: {sim:.3f})")
