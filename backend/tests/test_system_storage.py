import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.routers import system, history
from core.config import settings


def test_storage_reports_measured_files_and_restart_does_not_fake_success(tmp_path, monkeypatch):
    audio = tmp_path / 'audio'
    cache = tmp_path / 'cache'
    audio.mkdir(); cache.mkdir()
    (audio / 'control.wav').write_bytes(b'12345')
    (cache / 'control.bin').write_bytes(b'12')
    monkeypatch.setattr(settings, 'audio_output_dir', audio)
    monkeypatch.setattr(settings, 'cache_dir', cache)
    app = FastAPI(); app.include_router(system.router, prefix='/api')
    with TestClient(app) as client:
        data = client.get('/api/system/storage').json()
        assert data['audio_bytes'] == 5 and data['cache_bytes'] == 2
        assert data['disk_free_bytes'] > 0
        assert client.post('/api/system/restart-backend').status_code == 409


def test_actual_history_starred_query_contract():
    history.history_db.clear()
    item = history.add_history_entry('tts', 'Control', '/api/audio/control.wav', 1, 'en', 'system_speech')
    app = FastAPI(); app.include_router(history.router, prefix='/api')
    with TestClient(app) as client:
        assert client.patch(f'/api/history/{item.id}/star?starred=true').json()['starred'] is True
        assert client.patch(f'/api/history/{item.id}/star?starred=false').json()['starred'] is False
    history.history_db.clear()
