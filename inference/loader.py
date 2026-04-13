import boto3
import skops.io as sio
import json
from io import BytesIO

class S3Loader:
    def __init__(self, experiment):
        self.experiment = experiment
        self.s3 = boto3.client('s3')
        self.bucket = experiment['bucket']
        self.directory = experiment['directory']

    def load_model(self):
        # key = f"{self.directory}models/model.skops"
        # print(f"Downloading model from s3://{self.bucket}/{key}")
        #
        # response = self.s3.get_object(Bucket=self.bucket, Key=key)
        # model_file = BytesIO(response['Body'].read())
        untrusted_types = sio.get_untrusted_types(file='../training/artifacts/xg_signal_v1/20260412_171314/models/model.skops')
        # model = sio.load(model_file, trusted=untrusted_types)
        model = sio.load('../training/artifacts/xg_signal_v1/20260412_171314/models/model.skops', trusted=untrusted_types)
        return model

    def load_features(self):
        key = f"{self.directory}metadata.json"

        response = self.s3.get_object(Bucket=self.bucket, Key=key)

        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return data['feature_metadata']

    def load_config(self):
        key = f"{self.directory}config.json"

        response = self.s3.get_object(Bucket=self.bucket, Key=key)

        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return data

