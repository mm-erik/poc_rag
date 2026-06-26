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

	system_prompt = (
		"Du bist ein nahbarer und kompetenter Assistent, der Fragen auf Basis der bereitgestellten "
		"Kontextdokumente beantwortet. Antworte in natürlichem, fließendem und freundlichem Ton. "
		"Beantworte die Frage ausschließlich auf Grundlage des Kontexts. Wenn der Kontext nicht "
		"ausreicht, sage charmant und direkt, dass keine ausreichenden Informationen vorliegen. "
		"Erfinde keine Fakten, Zahlen oder Quellen. Antworte in der Sprache der Frage. "
		"Schreibe in ganzen, fließenden Sätzen ohne Listen oder Textdekorationen. Verzichte auf JEDE Textdekoration, wie z. B. Sternchen, Emojis oder Hervorhebungen. "
		"Gib als Ausgabe ausschließlich reinen Fließtext ohne Überschrift und ohne Präfix wie 'Antwort:'. "
		"Wenn mehrere Kontextabschnitte relevant sind, verbinde sie zu einer logischen Antwort."
	)

	user_prompt = (
		"Kontext:\n"
		+ "\n\n".join(context_lines)
		+ "\n\nFrage:\n"
		+ user_query
	)

	try:
		response = client.chat.complete(
			model=GENERATOR_MODEL,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.0,
			max_tokens=1000,
		)
		answer = _extract_text_content(response.choices[0].message.content).strip()
		return answer or "Ich konnte keine Antwort aus dem Kontext generieren."
	except Exception:
		return "Ich konnte aufgrund eines Modellfehlers keine Antwort generieren."