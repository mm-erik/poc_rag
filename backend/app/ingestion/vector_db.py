from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os
import app.ingestion.embed as embed
from uuid import uuid4
import app.utils.bm25 as bm25

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "text")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse-text")

qdrant = QdrantClient(url=QDRANT_URL)


def _extract_retrieval_content(payload: dict) -> str:
    element_type = payload.get("element_type", "")

    if element_type == "Image":
        return str(payload.get("image_description") or payload.get("text") or "").strip()

    if element_type == "Table":
        table_html = str(payload.get("table_as_html") or "").strip()
        table_description = str(payload.get("table_description") or "").strip()
        if table_html and table_description:
            return f"Table HTML:\n{table_html}\n\nTable Description:\n{table_description}"
        return table_description or table_html or str(payload.get("text") or "").strip()

    return str(payload.get("text") or "").strip()


def query_points(user_query: str, filters: list[qm.FieldCondition] = None) -> list[dict]:
    search_filter = qm.Filter(must=filters) if filters else None
    points = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qm.Prefetch(
                query=embed.get_embedding(user_query),
                using=DENSE_VECTOR_NAME,
                filter=search_filter,
                limit=20,
            ),
            qm.Prefetch(
                query=bm25.bm25_document(user_query),
                using=SPARSE_VECTOR_NAME,
                filter=search_filter,
                limit=20,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=search_filter,
        with_payload=True,
        limit=20,
    ).points

    parsed_hits: list[dict] = []
    for point in points:
        payload = point.payload or {}
        parsed_hits.append(
            {
                "score": point.score,
                "resource_id": payload.get("resource_id"),
                "title": payload.get("title"),
                "chunk_index": payload.get("chunk_index"),
                "element_type": payload.get("element_type"),
                "content": _extract_retrieval_content(payload),
            }
        )

    return parsed_hits

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
                    SPARSE_VECTOR_NAME: bm25.bm25_document(embedding_source),
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
