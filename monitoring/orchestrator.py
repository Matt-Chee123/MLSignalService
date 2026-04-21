from monitoring.config import BUCKET, AWS_REGION, MONITORED_MODELS
from monitoring.data_collector import DataCollector
from monitoring.drift_monitor import DriftMonitor
import monitoring.config

if __name__ == "__main__":
    print("Here")
    dataCollector = DataCollector(BUCKET, AWS_REGION, MONITORED_MODELS[0])
    config = dataCollector.load_config()
    metadata = dataCollector.load_metadata()
    reference = dataCollector.load_reference()
    predictions = dataCollector.load_predictions()

    driftMonitor = DriftMonitor(config, reference, predictions, metadata)
    feat_drift = driftMonitor.detect_feature_drift()
    pred_drift = driftMonitor.detect_prediction_drift()
    coverage_drift = driftMonitor.detect_coverage_drift()
    print(coverage_drift)
    print(feat_drift)
    print(pred_drift)