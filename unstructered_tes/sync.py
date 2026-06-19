import json
import mimetypes
import os
from pathlib import Path

import dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import PartitionRequest
from unstructured_client.models.shared import Files, PartitionParameters, Strategy

dotenv.load_dotenv()

DEFAULT_API_URL = "https://platform.unstructuredapp.io/api/v1"
DEFAULT_INPUT_DIR = "../input"
DEFAULT_OUTPUT_DIR = "../est"


def normalize_api_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")

    # Accept either base URL or accidentally provided /jobs URL.
    if url.endswith("/jobs"):
        url = url[: -len("/jobs")]

    if not url.endswith("/api/v1"):
        url = f"{url}/api/v1"

    return url


def partition_files_sync(client: UnstructuredClient, api_key: str) -> None:
    input_dir = Path(DEFAULT_INPUT_DIR).resolve()
    output_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = [path for path in sorted(input_dir.iterdir()) if path.is_file()]
    if not file_paths:
        raise RuntimeError(f"No files found in input directory: {input_dir}")

    for file_path in file_paths:
        print(f"Processing: {file_path.name}")

        with file_path.open("rb") as file_handle:
            content_type, _ = mimetypes.guess_type(str(file_path))

            response = client.general.partition(
                request=PartitionRequest(
                    partition_parameters=PartitionParameters(
                        files=Files(
                            content=file_handle,
                            file_name=file_path.name,
                            content_type=content_type or "application/octet-stream",
                        ),
                        strategy=Strategy.VLM,
                        vlm_model="gpt-4o",
                        vlm_model_provider="openai",
                        pdf_infer_table_structure=True,
                        include_page_breaks=True,
                        languages=["deu", "eng"],
                        extract_image_block_types=["Image", "Table"],
                    ),
                    unstructured_api_key=api_key,
                )
            )

        elements = response.elements
        if elements is None:
            raise RuntimeError(f"No partition output returned for file: {file_path.name}")

        output_path = output_dir / f"{file_path.stem}.json"
        output_path.write_text(json.dumps(elements, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is not set.")

    api_url = normalize_api_url(os.getenv("UNSTRUCTURED_API_URL", DEFAULT_API_URL))

    client = UnstructuredClient(api_key_auth=api_key, server_url=api_url)

    partition_files_sync(client, api_key)