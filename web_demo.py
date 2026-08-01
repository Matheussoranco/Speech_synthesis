import streamlit as st
from src.tts_model import TTSWrapper
from src.speaker_encoder import SpeakerEncoder
import numpy as np
import tempfile
import soundfile as sf

st.title("Speech Synthesis & Voice Cloning Demo")

mode = st.radio("Choose mode", ["TTS", "Voice Cloning"])
text = st.text_area("Text to Synthesize", "Hello, this is a demo.")


# Streamlit re-executes this whole script on every widget interaction, so
# constructing TTSWrapper()/SpeakerEncoder() unconditionally at module level
# reloaded the (potentially multi-GB) Coqui TTS model from scratch on every
# button click. Cache them across reruns instead.
@st.cache_resource
def load_tts():
    return TTSWrapper()


@st.cache_resource
def load_speaker_encoder():
    return SpeakerEncoder()


tts = load_tts()

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
        speaker_encoder = load_speaker_encoder()
        # SpeakerEncoder has no `extract_embedding` method; the real API is
        # embed_utterance_numpy(wav, sr).
        emb = speaker_encoder.embed_utterance_numpy(wav_data, sr)
        # tts.synthesize()'s speaker_wav is forwarded straight to Coqui TTS,
        # which requires a real file path — not the Streamlit UploadedFile
        # object `ref_audio`, which the original code passed directly and
        # which its `sf.read()` call above had already consumed anyway.
        ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(ref_tmp.name, wav_data, sr)
        wav = tts.synthesize(text, speaker_wav=ref_tmp.name)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tts.save(wav, tmp.name)
        st.audio(tmp.name)
