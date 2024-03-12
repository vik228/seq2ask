from __future__ import annotations

from env import bucket_name
from google.cloud import storage


def upload_to_gcs(filename, bucket_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(filename)
    print("file uploaded to GCP")
