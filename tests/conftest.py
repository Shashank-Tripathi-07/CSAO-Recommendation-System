import pytest
from fastapi.testclient import TestClient
from csao.api.main import app
import os

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_headers():
    return {"X-API-Key": "csao-secret-key-123"}
