# Backend (Quick and Dirty RAG PoC)

This backend is a minimal FastAPI service that:

- accepts text uploads at POST /upload
- chunks + hash-embeds text
- stores vectors in a Dockerized Qdrant instance
- searches tenant-scoped context at POST /chat

Dependency management is handled with uv via pyproject.toml.

## Endpoints

- GET /health
- POST /upload
- POST /chat

## Run with Docker Compose (from repo root)

docker compose up --build

API will be available at http://localhost:8000.
Qdrant will be available at http://localhost:6333.

## Stop

docker compose down

To also remove Qdrant data volume:

docker compose down -v
