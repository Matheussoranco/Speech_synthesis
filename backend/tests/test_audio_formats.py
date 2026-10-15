import io
import shutil
import sys
import wave
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.routers import openai_compat, dubbing, transcribe, audiobook
from core.config import settings

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'audio_output_dir', tmp_path)
    app = FastAPI()
    app.include_router(openai_compat.router)
    for router in (dubbing.router, transcribe.router, audiobook.router): app.include_router(router, prefix='/api')
    with TestClient(app) as client: yield client

@pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('ffmpeg'), reason='Real Windows speech and installed FFmpeg required')
@pytest.mark.parametrize('format,content_type,magic', [('mp3','audio/mpeg',b'ID3'), ('flac','audio/flac',b'fLaC'),
    ('opus','audio/ogg',b'OggS'), ('wav','audio/wav',b'RIFF'), ('aac','audio/aac',None), ('pcm','application/octet-stream',None)])
def test_real_encoded_audio_matches_its_declared_format(client, tmp_path, format, content_type, magic):
    response = client.post('/v1/audio/speech', json={'input': 'Este é um teste real de formato de áudio.',
        'voice': 'Microsoft Maria Desktop', 'response_format': format})
    assert response.status_code == 200, response.text
    assert response.headers['content-type'].startswith(content_type)
    assert len(response.content) > 1000
    if magic: assert response.content.startswith(magic)
    if format == 'wav':
        with wave.open(io.BytesIO(response.content)) as audio: assert audio.getnframes() > 1000
    if format == 'aac': assert response.content[0] == 0xff and response.content[1] & 0xf0 == 0xf0
    assert not list(tmp_path.glob('speech-*'))

@pytest.mark.parametrize('path', ['/api/dubbing/generate','/api/dubbing/batch','/api/transcribe/generate',
    '/api/audiobook/projects','/api/audiobook/projects/control/generate'])
def test_unimplemented_jobs_never_claim_started_or_completed(client, path):
    assert client.post(path).status_code == 503
    assert client.get('/api/transcribe/jobs/unknown').status_code == 404

def test_transcription_and_translation_do_not_invent_uploaded_audio_contents(client):
    for path in ('/v1/audio/transcriptions', '/v1/audio/translations'):
        response = client.post(path, files={'file': ('control.wav', b'not-real-audio', 'audio/wav')})
        assert response.status_code == 503
        assert 'mock' not in response.text.lower()
