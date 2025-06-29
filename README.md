# Speech Synthesis Project

This project aims to build an advanced speech synthesis model and a voice cloner, inspired by state-of-the-art LLM-based TTS systems.

## Structure
- `src/` - Main source code for training, inference, and utilities
- `models/` - Pretrained and custom model checkpoints
- `data/` - Datasets for training and evaluation
- `notebooks/` - Jupyter notebooks for experiments and prototyping

## Goals
- High-quality text-to-speech (TTS) synthesis
- Voice cloning from short audio samples
- Modular, extensible codebase

## References
- [Chatterbox](https://github.com/ChatterboxAI/chatterbox)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [YourTTS](https://github.com/Edresson/YourTTS)

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your dataset (see `src/data.py` for format)
3. Train a model: `python main.py train --config config.yaml --data data/ --output models/`
4. Synthesize speech: `python main.py infer --model models/model_x.pth --text "Hello world" --output output.wav`
5. Clone a voice: `python main.py clone --model models/model_x.pth --reference data/ref.wav --text "Hello" --output cloned.wav`
6. Run the web demo: `streamlit run web_demo.py`

## Features
- Real TTS model integration (Coqui TTS, Tacotron2, YourTTS)
- Neural vocoder (HiFi-GAN)
- Speaker encoder for voice cloning (Resemblyzer)
- Phonemization for robust text processing
- Web demo (Streamlit)
- Unit and integration tests

## References
- [Chatterbox](https://github.com/ChatterboxAI/chatterbox)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [YourTTS](https://github.com/Edresson/YourTTS)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
