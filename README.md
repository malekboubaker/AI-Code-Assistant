# AI Code Assistant

Local-first AI code assistant backend with model-independent agent orchestration and local RAG.

## Local Flow

VS Code / CLI / Gradio -> `http://localhost:8000` -> FastAPI -> Agent Orchestrator -> Qdrant Retriever -> Prompt Builder -> Local Model Provider -> Validator -> Formatter.

No external APIs are required. Ollama and Qdrant are local services.

## Start

```powershell
docker compose up -d qdrant
ollama pull qwen3:8b
ollama pull nomic-embed-text
pip install -r backend\requirements.txt
python scripts\start_backend.py
```

Index a small project:

```powershell
python scripts\index_project.py . --max-files 25
```

Generate through localhost:

```powershell
curl -X POST http://localhost:8000/api/v1/generate -H "Content-Type: application/json" -d "{\"instruction\":\"write a python fibonacci function\",\"language\":\"python\",\"use_rag\":true}"
```
