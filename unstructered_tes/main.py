import json
import mimetypes
import os
from pathlib import Path
import time

import dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import CreateJobRequest, DownloadJobOutputRequest
from unstructured_client.models.shared import BodyCreateJob, InputFiles

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


def run_on_demand_job(client: UnstructuredClient) -> tuple[str, list[str]]:
    input_files: list[InputFiles] = []
    opened_files = []
    input_dir = Path(DEFAULT_INPUT_DIR).resolve()

    try:
        for file_path in sorted(input_dir.iterdir()):
            if not file_path.is_file():
                continue

            content_type, _ = mimetypes.guess_type(str(file_path))
            file_handle = file_path.open("rb")
            opened_files.append(file_handle)
            input_files.append(
                InputFiles(
                    content=file_handle,
                    file_name=file_path.name,
                    content_type=content_type or "application/octet-stream",
                )
            )

        if not input_files:
            raise RuntimeError(f"No files found in input directory: {input_dir}")

        vlm_partitioner_node = {
            "name": "Partitioner",
            "subtype": "vlm",
            "type": "partition",
            "settings": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
                "is_dynamic": False,
                "allow_fast": True,
            },
        }

        image_description_enrichment_node = {
            "name": "Anthropic Image Description",
            "subtype": "anthropic_image_description",
            "type": "prompter",
            "settings": {},
        }

        table_description_enrichment_node = {
            "name": "Anthropic Table Description",
            "subtype": "anthropic_table_description",
            "type": "prompter",
            "settings": {},
        }

        job_nodes = [
            vlm_partitioner_node,
            image_description_enrichment_node,
            table_description_enrichment_node,
        ]

        response = client.jobs.create_job(
            request=CreateJobRequest(
                body_create_job=BodyCreateJob(
                    request_data=json.dumps({"job_nodes": job_nodes}),
                    input_files=input_files,
                )
            )
        )

        job_id = getattr(response.job_information, "id", None)
        if not job_id:
            raise RuntimeError("Transform API did not return a job ID.")

        input_file_ids = list(response.job_information.input_file_ids or [])
        if not input_file_ids:
            raise RuntimeError("Transform API did not return input file IDs.")

        return job_id, input_file_ids
    finally:
        for file_handle in opened_files:
            file_handle.close()


def poll_job_until_completed(client: UnstructuredClient, job_id: str) -> None:
    while True:
        response = client.jobs.get_job(request={"job_id": job_id})
        status = response.job_information.status
        print(f"Job status: {status}")

        if status == "COMPLETED":
            return

        if status in {"FAILED", "STOPPED"}:
            raise RuntimeError(f"Job did not complete successfully: {status}")

        time.sleep(10)


def _remove_image_base64_inplace(elements: list[dict]) -> None:
    for element in elements:
        metadata = element.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("image_base64", None)


def download_outputs(client: UnstructuredClient, job_id: str, input_file_ids: list[str]) -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_id in input_file_ids:
        print(f"Downloading processed output for file_id: {file_id}")
        response = client.jobs.download_job_output(
            request=DownloadJobOutputRequest(
                job_id=job_id,
                file_id=file_id,
            )
        )

        payload = response.any
        if isinstance(payload, list):
            _remove_image_base64_inplace(payload)

        output_path = output_dir / f"{file_id}.json"

        if payload is not None:
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved: {output_path}")
        else:
            raise RuntimeError(f"Failed to extract JSON content for file_id {file_id}")


if __name__ == "__main__":
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is not set.")

    api_url = normalize_api_url(os.getenv("UNSTRUCTURED_API_URL", DEFAULT_API_URL))

    client = UnstructuredClient(api_key_auth=api_key, server_url=api_url)

    job_id, input_file_ids = run_on_demand_job(client)
    print(f"Job ID: {job_id}")

    poll_job_until_completed(client, job_id)
    download_outputs(client, job_id, input_file_ids)