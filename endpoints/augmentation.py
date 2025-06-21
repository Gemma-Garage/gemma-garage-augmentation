from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
from google.cloud import storage

# The synthetic-data-kit is installed in the Docker container, so these imports will work there.
from synthetic_data_kit.core.ingest import process_file as ingest_process_file # type: ignore
from synthetic_data_kit.core.create import process_file as create_qa_process_file # type: ignore
from synthetic_data_kit.utils.config import load_config, get_llm_provider, get_openai_config, get_path_config # type: ignore
from synthetic_data_kit.core.context import AppContext # type: ignore

router = APIRouter()

class AugmentationRequest(BaseModel):
    file_name: str

def download_gcs_file(gcs_path: str, local_dir: str) -> str:
    """Downloads a file from GCS and returns the local file path."""
    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path format. It must start with 'gs://'.")
    
    bucket_name, *blob_parts = gcs_path[5:].split('/')
    source_blob_name = "/".join(blob_parts)
    
    if not source_blob_name:
        raise ValueError("Invalid GCS path format. Blob name is missing.")

    file_name = os.path.basename(source_blob_name)
    local_file_path = os.path.join(local_dir, file_name)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(local_dir, exist_ok=True)
    blob.download_to_filename(local_file_path)
    
    return local_file_path

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}.")

@router.post("/", tags=["Augmentation"])
async def augment_data(request: AugmentationRequest):
    """
    This endpoint downloads a file from a GCS path, processes it to extract text,
    generates question-answer pairs, and saves the result back to GCS.
    """
    ctx = AppContext()
    # Construct path to config.yaml relative to this file's location for robustness
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "synthetic-data-kit/synthetic_data_kit", "config.yaml")

    if not os.path.exists(config_path):
        raise HTTPException(status_code=500, detail=f"Configuration file not found at {config_path}")
    ctx.config = load_config(config_path)
    ctx.config_path = config_path

    bucket_uri = os.getenv("NEW_DATA_BUCKET")
    if not bucket_uri:
        raise HTTPException(status_code=500, detail="NEW_DATA_BUCKET environment variable not set")

    gcs_path = f"{bucket_uri}/{request.file_name}"
    bucket_name = bucket_uri.replace("gs://", "")

    temp_dir = "temp_processing"
    output_dir = "output"
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    local_file_path = None
    qa_output_path = None
    try:
        # 1. Download file from GCS
        local_file_path = download_gcs_file(gcs_path, temp_dir)
        file_name = os.path.basename(local_file_path)
        output_name = os.path.splitext(file_name)[0]

        # 2. Ingest and process the file to extract text
        parsed_file_path = ingest_process_file(local_file_path, output_dir, output_name, ctx.config)

        # 3. Generate QA pairs from the processed text
        provider = get_llm_provider(ctx.config)
        api_endpoint_config = get_openai_config(ctx.config)
        api_base = api_endpoint_config.get("api_base")
        model = api_endpoint_config.get("model")
        
        qa_output_path = create_qa_process_file(
            file_path=parsed_file_path,
            output_dir=output_dir,
            config_path=ctx.config_path,
            api_base=api_base,
            model=model,
            content_type="qa",
            num_pairs=10, # This could be a request parameter
            verbose=True,
            provider=provider
        )

        # 4. Upload generated QA pairs to GCS
        output_filename_base, _ = os.path.splitext(request.file_name)
        augmented_filename = f"{output_filename_base}_augmented.json"
        
        input_dir = os.path.dirname(request.file_name)
        if input_dir and input_dir != '.':
            destination_blob_name = f"{input_dir}/{augmented_filename}"
        else:
            destination_blob_name = augmented_filename

        upload_to_gcs(qa_output_path, bucket_name, destination_blob_name)
        
        destination_gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
        return {"message": f"Successfully augmented file and saved to {destination_gcs_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Cleanup the downloaded file and generated qa file
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
        if qa_output_path and os.path.exists(qa_output_path):
            os.remove(qa_output_path)