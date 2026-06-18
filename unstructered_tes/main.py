import json
import mimetypes
import os
from pathlib import Path

import dotenv
import requests
from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import CreateJobRequest
from unstructured_client.models.shared import BodyCreateJob, InputFiles

dotenv.load_dotenv()

DEFAULT_API_URL = "https://platform.unstructuredapp.io/api/v1"
DEFAULT_INPUT_DIR = "./input"
DEFAULT_OUTPUT_DIR = "./est"


def normalize_api_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")

    # Accept either base URL or accidentally provided /jobs URL.
    if url.endswith("/jobs"):
        url = url[: -len("/jobs")]

    if not url.endswith("/api/v1"):
        url = f"{url}/api/v1"

    return url


def create_transform_job(client: UnstructuredClient) -> str:
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

        response = client.jobs.create_job(
            request=CreateJobRequest(
                body_create_job=BodyCreateJob(
                    request_data=json.dumps(
                        {
                            "job_nodes": [
                                {
                                    "name": "Partitioner",
                                    "type": "partition",
                                    "subtype": "unstructured_api",
                                    "settings": {
                                        "strategy": "hi_res",
                                        "pdf_infer_table_structure": True,
                                        "include_page_breaks": True,
                                        "coordinates": True,
                                        "ocr_languages": ["deu", "eng"]
                                    },
                                }
                            ]
                        }
                    ),
                    input_files=input_files,
                )
            )
        )

        job_id = getattr(response.job_information, "id", None)
        if not job_id:
            raise RuntimeError("Transform API did not return a job ID.")
        return job_id
    finally:
        for file_handle in opened_files:
            file_handle.close()


def poll_for_output_file_ids(api_url: str, api_key: str, job_id: str) -> list[str]:
    headers = {
        "accept": "application/json",
        "unstructured-api-key": api_key,
    }
    status_url = f"{api_url.rstrip('/')}/jobs/{job_id}"

    while True:
        response = requests.get(status_url, headers=headers, timeout=30)
        response.raise_for_status()
        job = response.json()

        status = job.get("status")
        print(f"Job status: {status}")

        if status == "COMPLETED":
            print(f"Completed job response: {json.dumps(job, indent=2)}")
            output_file_ids = [
                item.get("file_id")
                for item in job.get("output_node_files", [])
                if item.get("file_id")
            ]
            if not output_file_ids:
                output_file_ids = job.get("input_file_ids", [])
            if not output_file_ids:
                raise RuntimeError("Job completed, but no output file IDs were returned.")
            return output_file_ids

        if status in {"FAILED", "STOPPED"}:
            raise RuntimeError(f"Job did not complete successfully: {status}")


def download_outputs(api_url: str, api_key: str, job_id: str, output_file_ids: list[str]) -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_url = f"{api_url.rstrip('/')}/jobs/{job_id}/download"

    for file_id in output_file_ids:
        response = requests.get(
            download_url,
            params={"file_id": file_id},
            headers={"unstructured-api-key": api_key},
            timeout=60,
        )
        response.raise_for_status()

        output_path = output_dir / f"{file_id}.json"
        output_path.write_bytes(response.content)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is not set.")

    api_url = normalize_api_url(os.getenv("UNSTRUCTURED_API_URL", DEFAULT_API_URL))

    client = UnstructuredClient(api_key_auth=api_key, server_url=api_url)

    job_id = create_transform_job(client)
    print(f"Job ID: {job_id}")

    output_file_ids = poll_for_output_file_ids(api_url, api_key, job_id)
    download_outputs(api_url, api_key, job_id, output_file_ids)