from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os
import app.ingestion.embed as embed
from uuid import uuid4
from app.utils.bm25 import bm25_document as bm25

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "text")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse-text")

qdrant = QdrantClient(url=QDRANT_URL)

def ensure_collection() -> None:
    try:
        exists = qdrant.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False

    if not exists:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)
            },
        )


def insert_points(chunks: list[dict], userId: str, title: str) -> tuple[str, int]:
    points: list[qm.PointStruct] = []
    resource_id = str(uuid4())

    for idx, chunk in enumerate(chunks):
        chunk_type = chunk.get("type", "")
        metadata = chunk.get("metadata", {})
        if chunk_type == "Table":
            embedding_source = metadata.get("table_description") or chunk.get("text", "")
        elif chunk_type == "Image":
            embedding_source = metadata.get("image_description") or chunk.get("text", "")
        else:
            embedding_source = chunk.get("text", "")

        points.append(
            qm.PointStruct(
                id=str(uuid4()),
                vector={
                    DENSE_VECTOR_NAME: embed.get_embedding(embedding_source),
                    SPARSE_VECTOR_NAME: bm25(embedding_source),
                },
                payload={
                    "user_id": userId,
                    "tenant_id": userId,
                    "resource_id": resource_id,
                    "title": title,
                    "chunk_index": idx,
                    "element_type": chunk_type,
                    "text": chunk.get("text", ""),
                    "image_description": metadata.get("image_description"),
                    "table_description": metadata.get("table_description"),
                    "table_as_html": metadata.get("text_as_html"),
                },
            )
        )

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return resource_id, len(points)
