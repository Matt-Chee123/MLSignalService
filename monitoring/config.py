import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get('AWS_REGION','eu-west-2')
BUCKET = os.environ['MONITORING_BUCKET']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
MODEL_ARTIFACTS_BUCKET = os.environ.get("MODEL_ARTIFACTS_BUCKET", "ml-signal-service")
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MONITORED_MODELS = ['3m-model']
MONITORING_BUCKET = os.environ["MONITORING_BUCKET"]
MODEL_ALIAS = "production"

PREDICTIONS_PREFIX = f"s3://{MONITORING_BUCKET}/predictions"
REPORTS_PREFIX = f"s3://{MONITORING_BUCKET}/reports"

def _artifact_uri(model_name):
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, MODEL_ALIAS)
    run = client.get_run(version.run_id)
    return run.info.artifact_uri

def get_model_config(model_name):
    return f"{_artifact_uri(model_name)}/analysis/config.json"

def get_model_reference(model_name):
    return f"{_artifact_uri(model_name)}/analysis/data/reference.parquet"

def get_metadata_url(model_name):
    return f"{_artifact_uri(model_name)}/analysis/metadata.json"