from monitoring.config import BUCKET, AWS_REGION, MONITORED_MODELS
from monitoring.data_collector import DataCollector
import monitoring.config

if __name__ == "__main__":
    print("Here")
    dataCollector = DataCollector(BUCKET, AWS_REGION, MONITORED_MODELS[0])
    dataCollector.load_config()
    dataCollector.load_reference()