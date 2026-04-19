from pathlib import Path
import os
import mlflow
import mlflow.sklearn
import tempfile
import json
from mlflow.models.signature import infer_signature

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.run_id = config['run_id']
        self.experiment_name = config.get('experiment_name', 'default_experiment')
        self.model_uri = None

        self.tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')

    def start(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_id)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        clean = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean, step=step)

    def log_model(self, model, X_full, preds, artifact_path="model"):
        signature = infer_signature(X_full, preds)
        mlflow.sklearn.log_model(model,  artifact_path=artifact_path, signature=signature, input_example=X_full.iloc[:5])

        run_id = mlflow.active_run().info.run_id
        self.model_uri = f"runs:/{run_id}/{artifact_path}"

    def register_model(self, model_name):
        if self.model_uri is None:
            raise RuntimeError("log_model must be called before register_model")
        mlflow.register_model(model_uri=self.model_uri, name=model_name)

    def log_artifact(self, local_path):
        mlflow.log_artifact(local_path)

    def end(self):
        mlflow.end_run()