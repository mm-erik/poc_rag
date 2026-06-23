from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os
import app.ingestion.embed as embed
from uuid import uuid4

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "text")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse-text")
BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
BM25_LANGUAGE = os.getenv("BM25_LANGUAGE", "english")

qdrant = QdrantClient(url=QDRANT_URL)

def query_points(user_query: str, filters: list[qm.FieldCondition] = None) -> list[dict]:
    search_filter = qm.Filter(must=filters) if filters else None
    return qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qm.Prefetch(
                query=embed.get_embedding(user_query),
                using=DENSE_VECTOR_NAME,
                filter=search_filter,
                limit=10,
            ),
            qm.Prefetch(
                query=bm25_document(user_query),
                using=SPARSE_VECTOR_NAME,
                filter=search_filter,
                limit=10,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=search_filter,
        with_payload=True,
        limit=5,
    ).points

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

def bm25_document(text: str) -> qm.Document:
    return qm.Document(
        text=text,
        model=BM25_MODEL,
        options={"language": BM25_LANGUAGE},
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

        print(f"Embedding source for chunk {idx} [{chunk_type}]: {embedding_source[:100]}...")

        points.append(
            qm.PointStruct(
                id=str(uuid4()),
                vector={
                    DENSE_VECTOR_NAME: embed.get_embedding(embedding_source),
                    SPARSE_VECTOR_NAME: bm25_document(embedding_source),
                },
                payload={
                    "user_id": userId,
                    "tenant_id": userId,
                    "resource_id": resource_id,
                    "title": title,
                    "chunk_index": idx,
                    "text": chunk.get("text", ""),
                },
            )
        )

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return resource_id, len(points)