from fastapi.testclient import TestClient
import pytest

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_recommend_unauthorized(client):
    response = client.post("/recommend", json={
        "user_id": "u1",
        "session_id": "s1",
        "cart_items": ["item1"],
        "context": {}
    })
    assert response.status_code == 500  # Generic exception handler for AuthenticationError or mapped status code
    assert "error_code" in response.json()
    assert response.json()["error_code"] == "AUTHENTICATION_ERROR"

def test_recommend_validation_error(client, auth_headers):
    response = client.post("/recommend", json={
        "user_id": "u1",
        # missing session_id
        "cart_items": [],
        "context": {}
    }, headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
