import io
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.routers import profiles as routes
from core.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'audio_output_dir', tmp_path / 'audio')
    routes.profiles_db.clear()
    app = FastAPI()
    app.include_router(routes.router, prefix='/api')
    with TestClient(app) as client:
        yield client
    routes.profiles_db.clear()


def create(client, name='Referência com espaços.webm', content=b'reference-control'):
    return client.post('/api/profiles', data={'name': 'Minha voz', 'ref_audio_name': name,
        'ref_text': 'Texto de referência', 'language': 'pt'},
        files={'ref_audio': ('independent-upload-name.webm', content, 'audio/webm')})


@pytest.mark.parametrize('name', ['../escape.wav', '..\\escape.wav', '/escape.wav',
    'C:\\outside.wav', '\\outside.wav', 'C:relative.wav', '\\\\server\\share.wav',
    'stream.wav:proof', 'voice.wav.', 'voice.wav ', '.', '..', 'NUL.wav', 'con.txt',
    'COM1.wav', 'lpt².wav', 'CONIN$.wav', 'control\x7f.wav', 'bad?.wav'])
def test_non_leaf_and_windows_aliases_rejected_before_disk_effects(client, name):
    assert create(client, name).status_code == 422
    assert not Path(settings.audio_output_dir).exists()
    assert not routes.profiles_db


def test_real_multipart_lifecycle_preserves_bytes_and_independent_names(client):
    response = create(client)
    assert response.status_code == 200
    profile = response.json()
    path = Path(profile['ref_audio_path'])
    assert path.parent.parent == Path(settings.audio_output_dir) / 'profiles'
    assert path.name == 'Referência com espaços.webm'
    assert path.read_bytes() == b'reference-control'
    audio = client.get(f"/api/profiles/{profile['id']}/audio")
    assert audio.status_code == 200 and audio.content == b'reference-control'
    assert audio.headers['content-type'] == 'audio/webm'
    assert client.get('/api/profiles').json() == [profile]
    assert client.get(f"/api/profiles/{profile['id']}").json() == profile
    updated = client.patch(f"/api/profiles/{profile['id']}", data={'name': 'Nova voz', 'instruct': 'Calma'})
    assert updated.status_code == 200 and updated.json()['name'] == 'Nova voz'
    assert client.patch(f"/api/profiles/{profile['id']}", data={'name': '  '}).status_code == 422
    assert client.delete(f"/api/profiles/{profile['id']}").status_code == 200
    assert not path.exists() and not path.parent.exists()
    assert client.get(f"/api/profiles/{profile['id']}/audio").status_code == 404


def test_literal_percent_name_remains_literal(client):
    result = create(client, '%2e%2e%2fcontrol.wav')
    assert result.status_code == 200
    assert Path(result.json()['ref_audio_path']).name == '%2e%2e%2fcontrol.wav'


@pytest.mark.parametrize('kind', ['escaped_path', 'escaped_name', 'invalid_id'])
def test_unsafe_stored_state_cannot_read_or_delete_outside_file(client, tmp_path, kind):
    profile = routes.Profile(**create(client).json())
    outside = tmp_path / 'outside-control.wav'
    outside.write_bytes(b'preserve-outside')
    if kind == 'escaped_path':
        profile.ref_audio_path = str(outside)
    elif kind == 'escaped_name':
        profile.ref_audio_name = '..\\outside-control.wav'
    else:
        profile.id = '..'
    routes.profiles_db[profile.id] = profile
    assert client.get(f'/api/profiles/{profile.id}/audio').status_code in (404, 409)
    assert client.delete(f'/api/profiles/{profile.id}').status_code in (404, 409)
    assert outside.read_bytes() == b'preserve-outside'
    assert profile.id in routes.profiles_db


def test_upload_limit_and_empty_file_leave_no_partial_profile(client, monkeypatch):
    monkeypatch.setattr(settings, 'max_profile_upload_bytes', 3)
    assert create(client, content=b'1234').status_code == 413
    assert create(client, content=b'').status_code == 422
    assert not list((Path(settings.audio_output_dir) / 'profiles').iterdir())
    assert not routes.profiles_db


@pytest.mark.asyncio
async def test_read_failure_removes_owned_partial_file(client):
    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError('controlled interrupted read')
            return super().read(2)
    upload = UploadFile(file=Broken(b'partial'), filename='upload.webm')
    with pytest.raises(OSError, match='interrupted read'):
        await routes.create_profile('Control', upload, 'safe.webm', 'Reference', '', 'en')
    assert not list((Path(settings.audio_output_dir) / 'profiles').iterdir())
    assert not routes.profiles_db


def test_existing_directory_is_not_reused_or_deleted(client, monkeypatch):
    identifier = uuid.uuid4()
    directory = Path(settings.audio_output_dir) / 'profiles' / str(identifier)
    directory.mkdir(parents=True)
    original = directory / 'original.wav'
    original.write_bytes(b'preserved')
    monkeypatch.setattr(routes.uuid, 'uuid4', lambda: identifier)
    with pytest.raises(FileExistsError):
        create(client)
    assert original.read_bytes() == b'preserved'


def test_delete_preserves_unrelated_file_in_profile_directory(client):
    profile = create(client).json()
    audio = Path(profile['ref_audio_path'])
    unrelated = audio.parent / 'user-notes.txt'
    unrelated.write_bytes(b'preserve-user-notes')
    assert client.delete(f"/api/profiles/{profile['id']}").status_code == 200
    assert not audio.exists()
    assert unrelated.read_bytes() == b'preserve-user-notes'
    assert profile['id'] not in routes.profiles_db
