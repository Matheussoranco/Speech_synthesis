import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import settings
from core.csrf import request_guard


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, 'remote_api_key', 'test-local-key')
    monkeypatch.setattr(settings, 'allowed_hosts', ['192.168.0.10'])
    app = FastAPI()
    app.middleware('http')(request_guard)
    app.add_api_route('/api/thing', lambda: {'ok': True}, methods=['GET', 'POST'])
    with TestClient(app, base_url='http://127.0.0.1:3900') as client:
        yield client


def test_loopback_without_browser_headers_is_allowed(client):
    assert client.post('/api/thing').status_code == 200
    assert client.get('/api/thing', headers={'Host': 'localhost:3900'}).status_code == 200


def test_rebound_host_is_rejected(client):
    assert client.get('/api/thing', headers={'Host': 'evil.example:3900'}).status_code == 400


def test_allowed_remote_host_requires_api_key(client):
    remote = {'Host': '192.168.0.10:3900'}
    assert client.get('/api/thing', headers=remote).status_code == 401
    authed = {**remote, 'Authorization': 'Bearer test-local-key'}
    assert client.get('/api/thing', headers=authed).status_code == 200


def test_cross_site_write_is_rejected(client):
    assert client.post('/api/thing', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/thing', headers={'Origin': 'null'}).status_code == 403
    assert client.post('/api/thing', headers={'Origin': 'http://localhost:3901'}).status_code == 200
    assert client.post('/api/thing', headers={'Origin': 'voicestudio://voicestudio'}).status_code == 200
    assert client.get('/api/thing', headers={'Origin': 'https://evil.example'}).status_code == 200
