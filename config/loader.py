import json
from pathlib import Path
from datetime import datetime, date, timedelta

def load_config():
    config_path = Path(__file__).parent / "baseConfig.json"

    with open(config_path) as f:
        config = json.load(f)

    config = resolve_dates(config)
    config['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = resolve_paths(config)
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

def resolve_paths(config):
    paths_cfg = config['data']

    paths_cfg['raw_data_path'] = Path(paths_cfg['raw_data_path']) / config['run_id']
    paths_cfg['processed_data_path'] = Path(paths_cfg['processed_data_path']) / config['run_id']
    paths_cfg['split_data_path'] = Path(paths_cfg['split_data_path']) / config['run_id']

    return config

