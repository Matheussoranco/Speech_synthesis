import streamlit as st
from src.tts_model import TTSWrapper
from src.speaker_encoder import SpeakerEncoder
import numpy as np
import tempfile
import soundfile as sf

st.title("Speech Synthesis & Voice Cloning Demo")

mode = st.radio("Choose mode", ["TTS", "Voice Cloning"])
text = st.text_area("Text to Synthesize", "Hello, this is a demo.")

tts = TTSWrapper()

if mode == "TTS":
    if st.button("Synthesize"):
        wav = tts.synthesize(text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tts.save(wav, tmp.name)
        st.audio(tmp.name)
else:
    ref_audio = st.file_uploader("Upload Reference Audio", type=["wav"])
    if st.button("Clone Voice") and ref_audio is not None:
        wav_data, sr = sf.read(ref_audio)
        speaker_encoder = SpeakerEncoder()
        emb = speaker_encoder.extract_embedding(wav_data, sr)
        wav = tts.synthesize(text, speaker_wav=ref_audio)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tts.save(wav, tmp.name)
        st.audio(tmp.name)
