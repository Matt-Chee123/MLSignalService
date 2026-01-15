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

INTERVAL = '1d'