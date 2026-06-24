import app.utils.get_mistral_client as get_mistral

client = get_mistral.get_mistral_client()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return response.data[0].embedding


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="mistral-embed", inputs=texts)
    return [item.embedding for item in response.data]