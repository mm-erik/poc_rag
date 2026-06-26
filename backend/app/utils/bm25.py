from qdrant_client.http import models as qm
import os

BM25_LANGUAGE = os.getenv("BM25_LANGUAGE", "english")
BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")

def bm25_document(text: str) -> qm.Document:
    return qm.Document(
        text=text,
        model=BM25_MODEL,
        options={"language": BM25_LANGUAGE},
    )