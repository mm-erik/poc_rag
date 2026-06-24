import os

import app.utils.get_mistral_client as get_mistral

client = get_mistral.get_mistral_client()

GENERATOR_MODEL = os.getenv("MISTRAL_GENERATOR_MODEL", "mistral-small-latest")


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


def generate_answer(user_query: str, hits: list[dict]) -> str:
	context_lines: list[str] = []
	for idx, hit in enumerate(hits[:5]):
		context = str(hit.get("content", "")).strip()
		if not context:
			continue
		context_lines.append(f"[{idx + 1}] {context}")

	if not context_lines:
		return "Ich konnte in den gefundenen Dokumenten keinen verwertbaren Kontext finden."

	prompt = (
		"Du bist ein hilfreicher RAG-Assistent.\n"
		"Beantworte die Nutzerfrage ausschließlich auf Basis des bereitgestellten Kontexts.\n"
		"Wenn der Kontext nicht ausreicht, sage das klar.\n"
		"Antworte auf Deutsch und präzise.\n\n"
		f"Nutzerfrage:\n{user_query}\n\n"
		"Kontext:\n"
		+ "\n\n".join(context_lines)
	)

	try:
		response = client.chat.complete(
			model=GENERATOR_MODEL,
			messages=[{"role": "user", "content": prompt}],
			temperature=0.2,
			max_tokens=700,
		)
		answer = _extract_text_content(response.choices[0].message.content).strip()
		return answer or "Ich konnte keine Antwort aus dem Kontext generieren."
	except Exception:
		return "Ich konnte aufgrund eines Modellfehlers keine Antwort generieren."