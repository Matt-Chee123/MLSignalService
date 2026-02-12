import datetime
from datetime import date, timedelta

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

RAW_DATA_PATH = '../data/raw/'
PROCESSED_DATA_PATH = '../data/processed/'

OUTPUT_PATH = '../data/datasets'

INTERVAL = '1d'

TRAINING_CONFIG = {
    "experiment_name": "rf_signal_v1",
    "data": {
        "dataset_path": "../data/datasets/run_20260212_210236"
    },
    "model": {
        "model_type": "random_forest",
        "hyperparams": {
            "n_estimators": 200,
            "max_depth": 6
        }
    },
    "training": {
        "output_dir": "./artifacts"
    },
    "metrics": [
        "mse",
        "r2",
        "rank_ic"
    ]
}