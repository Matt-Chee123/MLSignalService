import os
import mlflow
from mlflow.tracking import MlflowClient

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
model_name = os.environ["MODEL_NAME"]
alias = os.environ.get("MODEL_ALIAS", "production")

mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
mv = client.get_model_version_by_alias(name=model_name, alias=alias)
print(f"Resolved @{alias} -> version {mv.version} (run_id={mv.run_id})")

model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
print("Model loaded.")

if model.metadata.signature:
    print("Input schema:")
    for col in model.metadata.signature.inputs.inputs:
        print(f"  {col.name}: {col.type}")
else:
    print("No signature logged on this model version.")