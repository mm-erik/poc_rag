import os
from typing import Optional

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client.http import models as qm
from app.ingestion.vector_db import ensure_collection, query_points, insert_points
from app.ingestion.ingest import ingest
from app.generator.rerank import rerank_hits
from app.generator.generate import generate_answer


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
        return UploadResponse(resourceId="", chunksInserted=0, message="No chunks created")

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
    hits = rerank_hits(req.message, hits, top_k=5)

    if not hits:
        return ChatResponse(
            answer="I could not find relevant context for your tenant. Upload resources first.",
            sources=[],
        )

    sources: list[dict] = []

    for hit in hits:
        sources.append(
            {
                "score": hit.get("score"),
                "resource_id": hit.get("resource_id"),
                "title": hit.get("title"),
                "chunk_index": hit.get("chunk_index"),
                "element_type": hit.get("element_type"),
            }
        )

    answer = generate_answer(req.message, hits)

    return ChatResponse(answer=answer, sources=sources)
