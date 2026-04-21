import boto3
from monitoring.config import get_model_config, get_model_reference
import json
import mlflow.artifacts
import pandas as pd

class DataCollector:
    def __init__(self, bucket_name, region, model_name):
        self.region = region
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.s3 = boto3.client('s3')

    def load_config(self):
        config_url = get_model_config(self.model_name)
        config = mlflow.artifacts.load_dict(config_url)
        return config

    def load_reference(self):
        reference_url = get_model_reference(self.model_name)
        reference_parq = mlflow.artifacts.download_artifacts(artifact_uri=reference_url)
        reference = pd.read_parquet(reference_parq)


