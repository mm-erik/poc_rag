import json
import os
import time

from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import CreateJobRequest, DownloadJobOutputRequest
from unstructured_client.models.shared import BodyCreateJob, InputFiles

from app.ingestion.describe import enrich_elements

DEFAULT_API_URL = "https://platform.unstructuredapp.io/api/v1"


def normalize_api_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")

    # Accept either base URL or accidentally provided /jobs URL.
    if url.endswith("/jobs"):
        url = url[: -len("/jobs")]

    if not url.endswith("/api/v1"):
        url = f"{url}/api/v1"

    return url


def run_on_demand_job(client: UnstructuredClient, file: InputFiles) -> tuple[str, list[str]]:
    input_files = [file]

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

    # Target: ~400 tokens per chunk for optimal chatbot context (1 token ≈ 4 chars)
    # - new_after_n_chars (1500): Soft cap (~375 tokens). System naturally stops here if a layout boundary is found.
    # - max_characters (2000): Hard cap (~500 tokens). Acts as a buffer to keep long, cohesive paragraphs intact.
    # - combine_under_n_chars (500): Glues tiny text fragments together to prevent contextless "micro-chunks".
    # - isolate_table (True): Forces tables into clean, isolated chunks—ideal for our local LLM enrichment step.
    chunk_by_title_chunker_workflow_node = {
        "name": "Chunker",
        "subtype": "chunk_by_title",
        "type": "chunk",
        "settings": {
            "multipage_sections": True,
            "combine_under_n_chars": 500,
            "new_after_n_chars": 1500,
            "max_characters": 2000,
            "overlap": 200,
            "overlap_all": False,
            "isolate_table": True,
        }
    }

    job_nodes = [
        vlm_partitioner_node,
        chunk_by_title_chunker_workflow_node,
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


def poll_job_until_completed(client: UnstructuredClient, job_id: str) -> None:
    while True:
        response = client.jobs.get_job(request={"job_id": job_id})
        status = response.job_information.status
        if status == "COMPLETED":
            return

        print(f"Job status: {status}, polling again in 10 seconds...")
        time.sleep(10)

        if status in {"FAILED", "STOPPED"}:
            raise RuntimeError(f"Job did not complete successfully: {status}")


def _remove_image_base64_inplace(elements: list[dict]) -> None:
    for element in elements:
        metadata = element.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("image_base64", None)


def download_outputs(client: UnstructuredClient, job_id: str, input_file_ids: list[str]) -> list[dict]:
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
            print(f"Enriching elements for file_id: {file_id}")
            enrich_elements(payload)
            _remove_image_base64_inplace(payload)

        if payload is not None:
            return payload
        else:
            raise RuntimeError(f"Failed to extract JSON content for file_id {file_id}")

    raise RuntimeError("No output payload was returned for the provided input file IDs.")


def ingest(content: bytes, file_name: str, content_type: str = "application/pdf") -> list[dict]:
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is not set.")

    input_file =InputFiles(
        content=content,
        file_name=file_name,
        content_type=content_type
    )

    api_url = normalize_api_url(os.getenv("UNSTRUCTURED_API_URL", DEFAULT_API_URL))

    client = UnstructuredClient(api_key_auth=api_key, server_url=api_url)

    job_id, input_file_ids = run_on_demand_job(client, input_file)
    print(f"Job ID: {job_id}")

    poll_job_until_completed(client, job_id)
    return download_outputs(client, job_id, input_file_ids)
