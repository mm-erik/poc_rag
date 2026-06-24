import json

import app.utils.get_mistral_client as get_mistral

client = get_mistral.get_mistral_client()

RERANK_MODEL = "mistral-small-latest"


def _extract_text_content(message_content) -> str:
	if isinstance(message_content, str):
		return message_content

	if isinstance(message_content, list):
		parts: list[str] = []
		for item in message_content:
			if isinstance(item, dict) and item.get("type") == "text":
				parts.append(str(item.get("text", "")))
		return "\n".join(parts)

	return ""


def rerank_hits(user_query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
	if not hits:
		return []

	capped_hits = hits[:20]
	documents = []
	for idx, hit in enumerate(capped_hits):
		documents.append(
			{
				"index": idx,
				"resource_id": hit.get("resource_id"),
				"title": hit.get("title"),
				"chunk_index": hit.get("chunk_index"),
				"element_type": hit.get("element_type"),
				"content": hit.get("content", ""),
			}
		)

	prompt = (
		"You are a retrieval reranker.\n"
		"Given a user query and candidate passages, return ONLY valid JSON in the shape: "
		"{\"ranked_indices\": [int, ...]}.\n"
		"Rules:\n"
		"- Rank from most relevant to least relevant.\n"
		"- Include only provided indices.\n"
		"- Do not include explanations.\n"
		"- Return at most 5 indices.\n\n"
		f"Query:\n{user_query}\n\n"
		f"Candidates:\n{json.dumps(documents, ensure_ascii=False)}"
	)

	try:
		response = client.chat.complete(
			model=RERANK_MODEL,
			messages=[{"role": "user", "content": prompt}],
			temperature=0,
			max_tokens=256,
		)
		content = _extract_text_content(response.choices[0].message.content).strip()
		parsed = json.loads(content)
		ranked_indices = parsed.get("ranked_indices", [])

		ranked_hits: list[dict] = []
		seen: set[int] = set()
		for value in ranked_indices:
			if not isinstance(value, int):
				continue
			if value < 0 or value >= len(capped_hits):
				continue
			if value in seen:
				continue
			seen.add(value)
			ranked_hits.append(capped_hits[value])
			if len(ranked_hits) >= top_k:
				break

		if ranked_hits:
			return ranked_hits
	except Exception:
		# Fallback to retrieval score if rerank call or JSON parsing fails.
		pass

	return sorted(capped_hits, key=lambda item: item.get("score") or 0, reverse=True)[:top_k]