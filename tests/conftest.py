"""Shared pytest fixtures and environment probes.

Several parts of this project need libraries that are not installable with
pip because they are native, OS-level packages: espeak-ng for phonemization
and FFmpeg for audio decoding. Tests that genuinely exercise those paths are
skipped with an explicit reason when the dependency is absent, rather than
failing -- a missing system library is an environment fact, not a defect in
the code under test.

Install them to run the full suite:
    espeak-ng   winget install eSpeak-NG.eSpeak-NG
    FFmpeg      winget install Gyan.FFmpeg    (the "full-shared" build)
"""
import functools
import shutil

import pytest


@functools.lru_cache(maxsize=1)
def espeak_available() -> bool:
    """True when phonemizer can actually load an espeak backend.

    Probes via the project's own resolver, so a machine where espeak-ng is
    installed but PHONEMIZER_ESPEAK_LIBRARY is unset still counts as available
    -- ``espeak-ng.exe`` being on PATH is not sufficient on Windows, since
    phonemizer loads the DLL rather than shelling out to the executable.
    """
    try:
        from src.text_processor import _ensure_espeak_library

        return bool(_ensure_espeak_library())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    """True when torchaudio can decode a file (needs FFmpeg/torchcodec)."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        import torchcodec  # noqa: F401

        return True
    except Exception:
        return False


requires_espeak = pytest.mark.skipif(
    not espeak_available(),
    reason="espeak-ng is not installed on this system (see tests/conftest.py)",
)

requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(),
    reason="FFmpeg/torchcodec is not installed on this system (see tests/conftest.py)",
)
