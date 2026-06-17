import os

from unstructured_ingest.pipeline.pipeline import Pipeline
from unstructured_ingest.interfaces import ProcessorConfig
from unstructured_ingest.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig,
)
from unstructured_ingest.processes.partitioner import PartitionerConfig
from unstructured_ingest.processes.chunker import ChunkerConfig
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path="."),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            strategy="hi_res",
            ocr_languages=["deu"],
            partition_by_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_endpoint="https://api.unstructuredapp.io/general/v0/general",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
            },
        ),
        chunker_config=ChunkerConfig(
            chunking_strategy="by_title",
            chunk_max_characters=1000,
        ),
        uploader_config=LocalUploaderConfig(output_dir="../est"),
    ).run()