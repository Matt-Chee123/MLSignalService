import io
import os
import uuid
import logging
from datetime import datetime, timezone, date

import boto3
from botocore.config import Config
import pandas as pd

MONITORING_BUCKET = os.environ["MONITORING_BUCKET"]


class PredictionLogger:
    def __init__(self, model_name, model_version, run_id, feature_names):
        self.s3 = boto3.client('s3')
        self.date = date.today()
        self.model_name = model_name
        self.model_version = model_version
        self.run_id = run_id
        self.feature_names = feature_names

    def build_log(self, data, predictions, tickers):
        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Missing feature columns in inference data: {missing}")
        print(tickers)
        print(predictions)
        log = data[self.feature_names].copy()
        log["ticker"] = tickers
        log["prediction"] = predictions
        log["date"] = self.date
        log["model_name"] = self.model_name
        log["model_version"] = str(self.model_version)
        log["run_id"] = self.run_id
        log["logged_at"] = datetime.now(timezone.utc)
        return log

    def write_s3(self, log_df):
        key = (
            f"predictions/"
            f"{self.model_name}/"
            f"{self.date}/"
            f"predictions_{uuid.uuid4().hex[:8]}.parquet"
        )
        buffer = io.BytesIO()
        log_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.s3.put_object(
        Bucket=MONITORING_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
        )

