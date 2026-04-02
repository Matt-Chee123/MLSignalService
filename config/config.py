from datetime import datetime, date, timedelta
from pathlib import Path


TICKERS = [

    # Consumer Staples
    'PG',    # Procter & Gamble
    'KO',    # Coca-Cola
    'PEP',   # PepsiCo
    'WMT',   # Walmart
    'COST',  # Costco
    'MDLZ',  # Mondelez
    'CL',    # Colgate-Palmolive
    'MO',    # Altria
    'PM',    # Philip Morris
    'KMB',   # Kimberly-Clark

    # Healthcare
    'JNJ',   # Johnson & Johnson
    'UNH',   # UnitedHealth
    'PFE',   # Pfizer
    'MRK',   # Merck
    'ABBV',  # AbbVie
    'LLY',   # Eli Lilly
    'TMO',   # Thermo Fisher
    'DHR',   # Danaher
    'BMY',   # Bristol-Myers Squibb
    'AMGN',  # Amgen

    # Technology
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'NVDA',  # Nvidia
    'GOOGL', # Alphabet
    'META',  # Meta
    'ORCL',  # Oracle
    'ADBE',  # Adobe
    'CRM',   # Salesforce
    'CSCO',  # Cisco
    'INTC',  # Intel
    'TXN',   # Texas Instruments
    'AMD',   # AMD
    'QCOM',  # Qualcomm
    'AVGO',  # Broadcom
    'IBM',   # IBM

    # Financials
    'JPM',   # JPMorgan
    'BAC',   # Bank of America
    'WFC',   # Wells Fargo
    'C',     # Citigroup
    'GS',    # Goldman Sachs
    'MS',    # Morgan Stanley
    'BLK',   # BlackRock
    'SCHW',  # Charles Schwab
    'AXP',   # American Express
    'SPGI',  # S&P Global

    # Industrials
    'BA',    # Boeing
    'CAT',   # Caterpillar
    'HON',   # Honeywell
    'UPS',   # UPS
    'RTX',   # RTX
    'LMT',   # Lockheed Martin
    'DE',    # Deere
    'GE',    # General Electric
    'MMM',   # 3M
    'UNP',   # Union Pacific

    # Energy
    'XOM',   # ExxonMobil
    'CVX',   # Chevron
    'COP',   # ConocoPhillips
    'SLB',   # Schlumberger
    'EOG',   # EOG Resources
    'PSX',   # Phillips 66
    'MPC',   # Marathon Petroleum

    # Consumer Discretionary
    'AMZN',  # Amazon
    'TSLA',  # Tesla
    'HD',    # Home Depot
    'MCD',   # McDonald's
    'NKE',   # Nike
    'SBUX',  # Starbucks
    'LOW',   # Lowe’s
    'BKNG',  # Booking Holdings
    'TJX',   # TJX Companies
    'F',     # Ford

    # Communication Services
    'DIS',   # Disney
    'NFLX',  # Netflix
    'CMCSA', # Comcast
    'T',     # AT&T
    'VZ',    # Verizon
    'TMUS',  # T-Mobile
    'CHTR',  # Charter

    # Utilities
    'NEE',   # NextEra Energy
    'DUK',   # Duke Energy
    'SO',    # Southern Company
    'AEP',   # American Electric Power
    'EXC',   # Exelon

    # Materials
    'LIN',   # Linde
    'APD',   # Air Products
    'SHW',   # Sherwin-Williams
    'FCX',   # Freeport-McMoRan
    'NEM',   # Newmont

    # Real Estate
    'PLD',   # Prologis
    'AMT',   # American Tower
    'CCI',   # Crown Castle
    'SPG',   # Simon Property Group
    'EQIX'   # Equinix
]


START_DATE = date.today() - timedelta(days=5*365)
END_DATE = date.today()

RAW_DATA_PATH = Path('../data/raw/')
PROCESSED_DATA_PATH = Path('../data/processed/')
SPLIT_DATA_PATH = Path('../data/datasets/')

OUTPUT_PATH = '../data/datasets'

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

HORIZON = 3

INTERVAL = '1d'

# TRAINING_CONFIG = {
#     "experiment_name": "rf_signal_v1",
#     "run_id": RUN_ID,
#     "horizon": HORIZON,
#     "benchmark": '^GSPC',
#     "strategy": 'long_only',
#     "data": {
#         "dataset_path": "../data/datasets/run_20260215_192421"
#     },
#     "model": {
#         "model_type": "random_forest",
#         "hyperparams": {
#             "n_estimators": 250,
#             "max_depth": 6
#         }
#     },
#     "training": {
#         "output_dir": "../training/artifacts"
#     },
#     "metrics": [
#         "mse",
#         "r2",
#         "rank_ic"
#     ]
# }

TRAINING_CONFIG = {
    "experiment_name": "rf_signal_v3",
    "run_id": RUN_ID,
    "horizon": HORIZON,
    "tickers": TICKERS,
    "benchmark": '^GSPC',
    "strategy": 'long_only',
    "data": {
        "dataset_path": "../data/datasets/run_20260215_192421"
    },
    "model": {
        "model_type": "random_forest",
        "hyperparams": {
            "n_estimators": 500,
            "max_depth": None
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