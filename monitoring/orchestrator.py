from monitoring.config import BUCKET, AWS_REGION, MONITORED_MODELS
from monitoring.data_collector import DataHandler
from monitoring.drift_monitor import DriftMonitor

if __name__ == "__main__":
    print("Here")
    dataCollector = DataHandler(BUCKET, AWS_REGION, MONITORED_MODELS[0])
    config = dataCollector.load_config()
    metadata = dataCollector.load_metadata()
    reference = dataCollector.load_reference()
    predictions = dataCollector.load_predictions()

    driftMonitor = DriftMonitor(config, reference, predictions, metadata)
    feat_drift = driftMonitor.detect_feature_drift()
    pred_drift = driftMonitor.detect_prediction_drift()
    cov_drift = driftMonitor.detect_coverage_drift()
    alerts, status = driftMonitor.evaluate_drift_alerts(feat_drift, pred_drift, cov_drift)
    dataCollector.push_data_to_s3(feat_drift, pred_drift, cov_drift, alerts, status)