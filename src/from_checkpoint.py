"""Shared checkpoint helper — single place that builds SynthesizerTrn from config.

``infer.py`` and ``clone.py`` duplicated the ~25-line ``SynthesizerTrn(...)``
construction (and drifted once already); ``train.py`` builds it a third way.
Import this instead so hyperparameter defaults change in exactly one spot.
"""

from __future__ import annotations

from typing import Any, Mapping

from .model import SynthesizerTrn


def synthesizer_hparams(config: Any) -> dict:
    """Plain ``{name: value}`` model hyperparams from an OmegaConf/Dict config."""
    hp: Mapping[str, Any] = config.model.get("params", {})
    get = hp.get if hasattr(hp, "get") else (lambda k, d=None: hp[k] if k in hp else d)
    return {
        "n_vocab": get("n_vocab", 512),
        "spec_channels": get("spec_channels", 513),
        "segment_size": get("segment_size", 8192),
        "inter_channels": get("inter_channels", 192),
        "hidden_channels": get("hidden_channels", 192),
        "filter_channels": get("filter_channels", 768),
        "n_heads": get("n_heads", 2),
        "n_layers": get("n_layers", 6),
        "kernel_size": get("kernel_size", 3),
        "resblock": get("resblock", "1"),
        "resblock_kernel_sizes": tuple(get("resblock_kernel_sizes", [3, 7, 11])),
        "resblock_dilation_sizes": tuple(
            tuple(d) for d in get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        ),
        "upsample_rates": tuple(get("upsample_rates", [8, 8, 2, 2])),
        "upsample_initial_channel": get("upsample_initial_channel", 512),
        "upsample_kernel_sizes": tuple(get("upsample_kernel_sizes", [16, 16, 4, 4])),
        "n_speakers": get("n_speakers", 0),
        "gin_channels": get("gin_channels", 256),
        "use_sdp": get("use_sdp", True),
    }


def build_synthesizer(config: Any, p_dropout: float = 0.0) -> SynthesizerTrn:
    """Construct ``SynthesizerTrn`` from *config* (inference defaults)."""
    return SynthesizerTrn(p_dropout=p_dropout, **synthesizer_hparams(config))


def load_synthesizer_state(
    model: SynthesizerTrn,
    checkpoint_path: str,
    device: str = "cpu",
    *,
    strict: bool = False,
    expected_sha256: str | None = None,
) -> dict:
    """Load ``net_g``/``state_dict`` weights into *model*; return raw checkpoint."""
    from .utils import secure_torch_load

    ckpt = secure_torch_load(
        checkpoint_path, map_location=device, expected_sha256=expected_sha256
    )
    state = ckpt.get("net_g", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=strict)
    return ckpt


__all__ = ["build_synthesizer", "load_synthesizer_state", "synthesizer_hparams"]
