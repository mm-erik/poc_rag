from mistralai import Mistral
import os

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return response.data[0].embedding


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="mistral-embed", inputs=texts)
    return [item.embedding for item in response.data]