import json
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.schemas import GenerateRequest

client = TestClient(app)

def test_generate_stream(monkeypatch):
    import backend.api.routes as routes
    
    def mock_run_stream(request):
        yield 'data: ' + json.dumps({'type': 'stream_start', 'task': 'code_gen'}) + '\n\n'
        yield 'data: ' + json.dumps({'type': 'stream_token', 'content': 'print'}) + '\n\n'
        yield 'data: ' + json.dumps({'type': 'stream_token', 'content': '("hello")'}) + '\n\n'
        yield 'data: ' + json.dumps({'type': 'final', 'response': {'task': 'code_gen', 'language': 'python', 'generated_code': 'print("hello")', 'explanation': '', 'used_rag': False, 'rag_sources': [], 'validation': {'valid': True, 'warnings': [], 'errors': [], 'duration_ms': 0}, 'metadata': {}}}) + '\n\n'
        
    routes.orchestrator.run_stream = mock_run_stream

    req = GenerateRequest(
        instruction="write hello world",
        code="",
        language="python",
        use_rag=False
    )
    
    response = client.post("/api/v1/generate_stream", json=req.model_dump())
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    lines = [line for line in response.text.split("\n\n") if line]
    assert len(lines) == 4
    
    start_event = json.loads(lines[0][6:])
    assert start_event["type"] == "stream_start"
    
    token1 = json.loads(lines[1][6:])
    assert token1["type"] == "stream_token"
    assert token1["content"] == "print"
    
    token2 = json.loads(lines[2][6:])
    assert token2["type"] == "stream_token"
    assert token2["content"] == "(\"hello\")"
    
    final_event = json.loads(lines[3][6:])
    assert final_event["type"] == "final"
    assert final_event["response"]["generated_code"] == "print(\"hello\")"

