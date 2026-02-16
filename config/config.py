from datetime import datetime, date, timedelta
from pathlib import Path


TICKERS = [
    'PG',    # Procter & Gamble
    'KO',    # Coca-Cola
    'WMT',   # Walmart
    'COST',  # Costco
    'JNJ',   # Johnson & Johnson
    'UNH',   # UnitedHealth
    'PFE',   # Pfizer
    'MRK'    # Merck
]


START_DATE = date.today() - timedelta(days=5*365)
END_DATE = date.today()

RAW_DATA_PATH = Path('../data/raw/')
PROCESSED_DATA_PATH = Path('../data/processed/')
SPLIT_DATA_PATH = Path('../data/datasets/')

OUTPUT_PATH = '../data/datasets'

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

INTERVAL = '1d'

TRAINING_CONFIG = {
    "experiment_name": "rf_signal_v1",
    "data": {
        "dataset_path": "../data/datasets/run_20260215_192421"
    },
    "model": {
        "model_type": "random_forest",
        "hyperparams": {
            "n_estimators": 200,
            "max_depth": 6
        }
    },
    "training": {
        "output_dir": "../training/artifacts"
    },
    "metrics": [
        "mse",
        "r2",
        "rank_ic"
    ]
}