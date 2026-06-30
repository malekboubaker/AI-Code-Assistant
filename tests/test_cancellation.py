import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.api import cancellation

client = TestClient(app)

def test_cancellation_registry():
    response_id = "test-123"
    assert not cancellation.is_cancelled(response_id)
    
    # Register event
    event = cancellation.register_request(response_id)
    assert not event.is_set()
    assert not cancellation.is_cancelled(response_id)
    
    # Cancel via API
    response = client.post(f"/api/v1/cancel/{response_id}")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Request test-123 cancelled."}
    
    assert event.is_set()
    assert cancellation.is_cancelled(response_id)
    
    # Cleanup
    cancellation.cleanup_request(response_id)
    assert not cancellation.is_cancelled(response_id)

def test_cancel_nonexistent_request():
    response = client.post("/api/v1/cancel/does-not-exist")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"
