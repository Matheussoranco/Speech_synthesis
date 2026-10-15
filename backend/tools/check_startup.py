"""Import and enter the real backend lifespan, without downloading a model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from main import app

with TestClient(app, base_url="http://127.0.0.1:3900") as client:
    response = client.get('/api/system/info')
    response.raise_for_status()
    assert response.json()['app_version']
print('Backend lifespan and local system info endpoint passed.')
