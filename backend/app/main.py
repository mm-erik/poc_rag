import os
from typing import Optional

from fastapi import FastAPI, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client.http import models as qm
from app.ingestion.vector_db import ensure_collection, query_points, insert_points
from app.ingestion.ingest import ingest


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "text")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse-text")


class UploadRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class UploadResponse(BaseModel):
    resourceId: str
    chunksInserted: int
    message: str


class ChatRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    filterResourceIds: Optional[list[str]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


app = FastAPI(title="RAG PoC Backend", version="0.1.0")

cors_origins = [x.strip() for x in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_collection()


@app.get("/")
def root() -> dict:
    return {"name": "RAG PoC Backend", "status": "ok"}


@app.get("/health")
def health() -> dict:
    try:
        ensure_collection()
        return {"status": "ok", "qdrant": "reachable"}
    except Exception as exc:
        return {"status": "error", "qdrant": "unreachable", "detail": str(exc)}


@app.post("/upload", response_model=UploadResponse)
async def upload_resource(
    userId: str = Form(...),
    title: str = Form(...),
    file: UploadFile = UploadFile(...),
) -> UploadResponse:
    ensure_collection()

    raw = await file.read()

    chunks = ingest(raw, file.filename, file.content_type)

    if not chunks:
        return UploadResponse(chunksInserted=0, message="No chunks created")

    resource_id, chunk_count = insert_points(chunks, userId, title)

    return UploadResponse(
        resourceId=resource_id,
        chunksInserted=chunk_count,
        message="Resource uploaded to Qdrant",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    ensure_collection()

    filters = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=req.userId)),
    ]

    if req.filterResourceIds:
        filters.append(
            qm.FieldCondition(
                key="resource_id",
                match=qm.MatchAny(any=req.filterResourceIds),
            )
        )

    hits = query_points(req.message, filters)

    if not hits:
        return ChatResponse(
            answer="I could not find relevant context for your tenant. Upload resources first.",
            sources=[],
        )

    snippets: list[str] = []
    sources: list[dict] = []

    for hit in hits:
        payload = hit.payload or {}
        snippet = str(payload.get("text", "")).strip()
        if snippet:
            snippets.append(snippet)

        sources.append(
            {
                "score": hit.score,
                "resource_id": payload.get("resource_id"),
                "title": payload.get("title"),
                "chunk_index": payload.get("chunk_index"),
            }
        )

    context_block = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets[:3]))
    answer = (
        "Quick PoC answer generated from retrieved Qdrant context. "
        "Replace this with your LLM call later.\n\n"
        f"User question: {req.message}\n\n"
        "Retrieved context:\n"
        f"{context_block}"
    )

    return ChatResponse(answer=answer, sources=sources)
