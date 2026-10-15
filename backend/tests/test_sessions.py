import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.routers import auth as routes
from core.auth import PrincipalKind, principal_for, sessions
from core.config import settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, 'remote_api_key', 'test-local-key')
    sessions.clear()
    app = FastAPI()
    app.include_router(routes.router, prefix='/api')
    with TestClient(app) as client:
        yield client
    sessions.clear()


def issue(client, transport='cookie'):
    return client.post('/api/auth/session', json={'transport': transport}, headers={'Authorization': 'Bearer test-local-key'})


def test_real_cookie_issue_inspection_and_revocation(client):
    response = issue(client)
    assert response.status_code == 204
    cookie = client.cookies.get('ov_session')
    assert cookie in sessions
    assert 'HttpOnly' in response.headers['set-cookie']
    assert client.get('/api/auth/session').json()['valid'] is True
    response = client.delete('/api/auth/session', headers={'Origin': 'http://localhost:3901'})
    assert response.status_code == 204
    assert 'Max-Age=0' in response.headers['set-cookie']
    assert cookie not in sessions
    assert client.cookies.get('ov_session') is None
    assert client.get('/api/auth/session').json() == {'valid': False}


def test_unknown_and_expired_cookies_do_not_become_admin(client):
    client.cookies.set('ov_session', 'forged')
    assert client.get('/api/auth/session').json() == {'valid': False}
    scope = {'type': 'http', 'method': 'GET', 'path': '/', 'headers': [(b'cookie', b'ov_session=forged')]}
    assert principal_for(Request(scope)).kind == PrincipalKind.ANONYMOUS
    sessions['forged'] = {'created': 0, 'expires': time.time() - 1}
    assert client.get('/api/auth/session').json() == {'valid': False}
    assert 'forged' not in sessions


def test_issued_bearer_inspection_logout_and_expiry(client):
    response = issue(client, 'bearer')
    assert response.status_code == 200
    token = response.json()['token']
    headers = {'Authorization': f'Bearer {token}'}
    assert client.get('/api/auth/session', headers=headers).json()['valid'] is True
    assert client.delete('/api/auth/session', headers=headers).status_code == 204
    assert client.get('/api/auth/session', headers=headers).json() == {'valid': False}
    sessions[token] = {'created': 0, 'expires': time.time() - 1}
    assert client.get('/api/auth/session', headers=headers).json() == {'valid': False}


def test_invalid_header_never_falls_back_to_cookie(client):
    issue(client)
    assert client.get('/api/auth/session', headers={'Authorization': 'Bearer wrong'}).json() == {'valid': False}
    assert client.post('/api/auth/session', json={'transport': 'cookie'}, headers={'Authorization': 'Bearer wrong'}).status_code == 401


@pytest.mark.parametrize('origin', ['http://localhost.attacker.example', 'http://localhost:3901.evil',
    'voicestudio://voicestudio.attacker', 'voicestudio://user@voicestudio', 'null', 'http://localhost:3903'])
def test_cookie_mutations_reject_foreign_origins_even_with_good_referer(client, origin):
    issue(client)
    cookie = client.cookies.get('ov_session')
    response = client.delete('/api/auth/session', headers={'Origin': origin, 'Referer': 'http://localhost:3901/clone'})
    assert response.status_code == 403
    assert cookie in sessions


@pytest.mark.parametrize('headers', [{'Origin': 'voicestudio://voicestudio'},
    {'Origin': 'http://localhost:3902'}, {'Referer': 'http://localhost:3901/settings'}])
def test_legacy_cookie_migration_from_allowed_origins(client, headers):
    client.cookies.set('ov_key', 'test-local-key')
    response = client.post('/api/auth/session', json={'transport': 'cookie'}, headers=headers)
    assert response.status_code == 204
    assert 'ov_session=' in response.headers['set-cookie']
    assert 'ov_key=""' in response.headers['set-cookie']


def test_cookie_origin_required_but_bearer_clients_need_no_origin(client):
    client.cookies.set('ov_key', 'test-local-key')
    assert client.post('/api/auth/session', json={'transport': 'cookie'}).status_code == 403
    assert issue(client, 'bearer').status_code == 200


@pytest.mark.parametrize('master', ['legacy', 'header'])
def test_logout_revokes_browser_session_with_another_master_credential(client, master):
    issue(client)
    token = client.cookies.get('ov_session')
    if master == 'legacy':
        client.cookies.set('ov_key', 'test-local-key')
        headers = {'Origin': 'http://localhost:3901'}
    else:
        headers = {'Authorization': 'Bearer test-local-key'}
    assert client.delete('/api/auth/session', headers=headers).status_code == 204
    assert token not in sessions
    assert client.get('/api/auth/session', headers={'Authorization': f'Bearer {token}'}).json() == {'valid': False}


def test_invalid_header_cannot_revoke_a_valid_cookie_session(client):
    issue(client)
    token = client.cookies.get('ov_session')
    response = client.delete('/api/auth/session', headers={'Authorization': 'Bearer wrong'})
    assert response.status_code == 401
    assert token in sessions
    assert client.cookies.get('ov_session') == token
