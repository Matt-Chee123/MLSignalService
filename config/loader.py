import os
import json
from pathlib import Path
from datetime import date, timedelta

def load_config():
    config_path = "./baseConfig.json"

    with open(config_path) as f:
        config = json.load(f)

    config = resolve_dates(config)
    return config

def resolve_dates(config):
    data_cfg = config['data']

    if data_cfg['end_date'] == 'today':
        end_date = date.today()
    else:
        end_date = date.fromisoformat(data_cfg['end_date'])

    start_date = end_date - timedelta(days=data_cfg['lookback_days'])

    data_cfg['start_date'] = start_date
    data_cfg['end_date'] = end_date

    return config

