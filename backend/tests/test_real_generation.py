import sys
import wave
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.routers import generate, engines, history
from core.config import settings

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'audio_output_dir', tmp_path)
    history.history_db.clear()
    app = FastAPI()
    for router in (generate.router, engines.router, history.router): app.include_router(router, prefix='/api')
    with TestClient(app) as client: yield client

def test_clone_and_design_never_emit_fake_success_or_audio(client, tmp_path):
    clone = client.post('/api/generate/clone', data={'text': 'Teste de voz', 'profile_id': 'control'})
    design = client.post('/api/generate/design', data={'text': 'Teste de voz', 'instruct': 'Calma'})
    assert clone.status_code == design.status_code == 503
    assert list(tmp_path.iterdir()) == []
    assert client.get('/api/history').json() == []

def test_unknown_models_cannot_be_marked_available(client):
    assert client.post('/api/engines/omnivoice/load').status_code == 503
    assert client.post('/api/engines/whisperx/load').status_code == 503
    response = client.get('/api/engines').json()
    assert response['asr'] == response['vocoder'] == []
    assert all('voice_cloning' not in item['capabilities'] for item in response['tts'])

@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows installed voice integration')
def test_real_installed_portuguese_voice_produces_valid_speech_wav_and_history(client, tmp_path):
    engine = client.get('/api/engines').json()['tts'][0]
    assert engine['status'] == 'available'
    assert 'pt-BR' in engine['languages']
    text = 'Este arquivo contém fala real em português, gerada pela voz instalada no computador.'
    response = client.post('/api/generate/tts', data={'text': text, 'language': 'pt'})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['engine'] == 'system_speech'
    assert result['language'] == 'pt-BR'
    assert result['duration'] > 1
    output = tmp_path / Path(result['audio_path']).name
    with wave.open(str(output)) as audio:
        assert audio.getnframes() > 1000
        assert audio.getframerate() == result['sample_rate']
    saved = client.get('/api/history').json()
    assert len(saved) == 1 and saved[0]['text'] == text
    assert saved[0]['metadata']['voice'] in engine['models']

def test_invalid_tts_controls_fail_without_audio(client, tmp_path):
    for data in ({'text': ' '}, {'text': 'a' * 5001}, {'text': 'Teste', 'speed': 'nan'},
        {'text': 'Teste', 'speed': 50}, {'text': 'Teste', 'pitch': 1.5}):
        assert client.post('/api/generate/tts', data=data).status_code == 422
    assert list(tmp_path.iterdir()) == []
