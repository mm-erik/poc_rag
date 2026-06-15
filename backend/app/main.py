import hashlib
import math
import os
import re
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "128"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))


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
    filterResourceId: Optional[str] = None


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

qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    try:
        exists = qdrant.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False

    if not exists:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE),
        )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start += step

    return chunks


def hash_embed(text: str, size: int = VECTOR_SIZE) -> list[float]:
    vec = [0.0] * size
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())

    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


@app.on_event("startup")
def startup() -> None:
    ensure_collection()


@app.get("/")
def root() -> dict:
    return {"name": "RAG PoC Backend", "status": "ok", "collection": QDRANT_COLLECTION}


@app.get("/health")
def health() -> dict:
    try:
        ensure_collection()
        return {"status": "ok", "qdrant": "reachable", "collection": QDRANT_COLLECTION}
    except Exception as exc:
        return {"status": "error", "qdrant": "unreachable", "detail": str(exc)}


@app.post("/upload", response_model=UploadResponse)
def upload_resource(req: UploadRequest) -> UploadResponse:
    ensure_collection()

    resource_id = str(uuid4())
    chunks = chunk_text(req.text)

    if not chunks:
        return UploadResponse(resourceId=resource_id, chunksInserted=0, message="No chunks created")

    points: list[qm.PointStruct] = []
    for idx, chunk in enumerate(chunks):
        points.append(
            qm.PointStruct(
                id=str(uuid4()),
                vector=hash_embed(chunk),
                payload={
                    "user_id": req.userId,
                    "tenant_id": req.userId,
                    "resource_id": resource_id,
                    "title": req.title,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
        )

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

    return UploadResponse(
        resourceId=resource_id,
        chunksInserted=len(points),
        message="Resource uploaded to Qdrant",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    ensure_collection()

    filters = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=req.userId)),
    ]

    if req.filterResourceId:
        filters.append(
            qm.FieldCondition(
                key="resource_id",
                match=qm.MatchValue(value=req.filterResourceId),
            )
        )

    search_filter = qm.Filter(must=filters)

    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=hash_embed(req.message),
        query_filter=search_filter,
        with_payload=True,
        limit=5,
    )

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
