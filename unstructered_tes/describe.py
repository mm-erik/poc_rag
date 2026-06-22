import os
import time
import dotenv

import requests

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

VISION_MODEL = "pixtral-12b-2409"
TEXT_MODEL = "mistral-small-latest"

REQUEST_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.5


def _post(payload: dict) -> dict:
    response = requests.post(
        MISTRAL_CHAT_URL,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def describe_image(image_base64: str, mime_type: str = "image/jpeg") -> str:
    result = _post({
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{image_base64}",
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe the content of this image concisely. "
                            "Focus on the key information it conveys."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 512,
    })
    return result["choices"][0]["message"]["content"].strip()


def describe_table(text_as_html: str) -> str:
    result = _post({
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the following HTML table concisely. "
                    "Describe what it shows and highlight the key data points:\n\n"
                    + text_as_html
                ),
            }
        ],
        "max_tokens": 512,
    })
    return result["choices"][0]["message"]["content"].strip()


def enrich_elements(elements: list[dict]) -> list[dict]:
    images_done = 0
    tables_done = 0

    for idx, element in enumerate(elements):
        element_type = element.get("type")
        metadata = element.setdefault("metadata", {})
        page = metadata.get("page_number", "?")

        if element_type == "Image":
            image_base64 = metadata.get("image_base64")
            if not image_base64:
                continue
            mime_type = metadata.get("image_mime_type", "image/jpeg")
            print(f"  [image] element {idx}, page {page} – describing via Mistral...")
            try:
                metadata["image_description"] = describe_image(image_base64, mime_type)
                images_done += 1
            except requests.HTTPError as exc:
                print(f"    Warning: HTTP {exc.response.status_code}: {exc.response.text[:200]}")
            except Exception as exc:
                print(f"    Warning: {exc}")
            time.sleep(RATE_LIMIT_SLEEP)

        elif element_type == "TableChunk":
            text_as_html = metadata.get("text_as_html")
            if not text_as_html:
                continue
            print(f"  [table] element {idx}, page {page} – describing via Mistral...")
            try:
                metadata["table_description"] = describe_table(text_as_html)
                tables_done += 1
            except requests.HTTPError as exc:
                print(f"    Warning: HTTP {exc.response.status_code}: {exc.response.text[:200]}")
            except Exception as exc:
                print(f"    Warning: {exc}")
            time.sleep(RATE_LIMIT_SLEEP)

    print(f"  Enrichment complete: {images_done} image(s), {tables_done} table(s) described.")
    return elements
