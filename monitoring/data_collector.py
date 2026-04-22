import boto3
from monitoring.config import get_model_config, get_model_reference, get_metadata_url
import mlflow.artifacts
import pandas as pd
from datetime import date, timedelta
from io import BytesIO
import uuid
import json

class DataHandler:
    def __init__(self, bucket_name, region, model_name, n_lookback=30):
        self.region = region
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.s3 = boto3.client('s3')
        self.lookback = n_lookback

    def load_config(self):
        config_url = get_model_config(self.model_name)
        config = mlflow.artifacts.load_dict(config_url)
        return config

    def load_metadata(self):
        metadata_url = get_metadata_url(self.model_name)
        metadata = mlflow.artifacts.load_dict(metadata_url)
        return metadata

    def load_reference(self):
        reference_url = get_model_reference(self.model_name)
        reference_parq = mlflow.artifacts.download_artifacts(artifact_uri=reference_url)
        reference = pd.read_parquet(reference_parq)
        return reference

    def load_predictions(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback)

        daily_frames = []
        current = start_date
        while current <= end_date:
            df = self.pull_data_for_date(current)
            if not df.empty:
                daily_frames.append(df)
            current += timedelta(days=1)

        if not daily_frames:
            raise ValueError(f"No predictions found between {start_date} and {end_date}")

        return pd.concat(daily_frames, ignore_index=True)

    def pull_data_for_date(self, target_date):
        prefix = f"predictions/model_name={self.model_name}/date={target_date}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        dfs = []
        for page in pages:
            for obj in page.get('Contents', []):
                if not obj['Key'].endswith('parquet'):
                    continue
                response = self.s3.get_object(Bucket=self.bucket_name, Key=obj['Key'])
                df = pd.read_parquet(BytesIO(response['Body'].read()))
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def push_data_to_s3(self, feature_drift, prediction_drift, coverage_drift, alerts, status):
        run_timestamp = date.today()
        run_date = run_timestamp.strftime("%Y-%m-%d")
        run_id = f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

        payload = {
            "run_id": run_id,
            "run_timestamp": run_timestamp.isoformat(),
            "model_name": self.model_name,
            "status": status,
            "feature_drift": feature_drift,
            "prediction_drift": prediction_drift,
            "coverage_drift": coverage_drift,
            "alerts": alerts,
        }

        key = f"monitoring/{self.model_name}/drift_runs/date={run_date}/{run_id}.json"

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(payload, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return key