from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os
from app.utils.bm25 import bm25_document as bm25
import app.ingestion.embed as embed

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
                query=bm25(user_query),
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