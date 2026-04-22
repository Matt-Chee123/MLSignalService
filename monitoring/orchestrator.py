from monitoring.config import BUCKET, AWS_REGION, MONITORED_MODELS
from monitoring.data_collector import DataHandler
from monitoring.drift_monitor import DriftMonitor

def run_monitoring(model_name):
    handler = DataHandler(BUCKET, AWS_REGION, model_name)
    config = handler.load_config()
    metadata = handler.load_metadata()
    reference = handler.load_reference()
    current = handler.load_predictions()

    monitor = DriftMonitor(config, reference, current, metadata, model_name)
    payload = monitor.run()
    key = handler.push_data_to_s3(payload)

    return {
        "model_name": model_name,
        "status": payload["status"],
        "alert_count": len(payload["alerts"]),
        "s3_key": key,
    }


def lambda_handler(event, context):
    model_name = event.get("model_name") or MONITORED_MODELS[0]
    result = run_monitoring(model_name)
    return {"statusCode": 200, **result}


if __name__ == "__main__":
    for model in MONITORED_MODELS:
        print(run_monitoring(model))