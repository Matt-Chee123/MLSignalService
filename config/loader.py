import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta

RUN_ID_PATH = Path("/data/current_run_id.txt")


def load_config():
    config_path = Path(__file__).parent / "baseConfig.json"
    with open(config_path) as f:
        config = json.load(f)

    config = resolve_dates(config)

    role = os.getenv("ROLE", "PRODUCER")

    if role == "PRODUCER":
        config['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(RUN_ID_PATH, "w") as f:
            f.write(config['run_id'])
    else:
        with open(RUN_ID_PATH, "r") as f:
            config['run_id'] = f.read().strip()

    config = resolve_hyperparams(config)
    config = resolve_paths(config)
    return config

def resolve_dates(config):
    data_cfg = config['data']

    if data_cfg['end_date'] == 'today':
        end_date = date.today()
    else:
        end_date = date.fromisoformat(data_cfg['end_date'])

    start_date = end_date - timedelta(days=data_cfg['lookback_days'])

    data_cfg['start_date'] = str(start_date)
    data_cfg['end_date'] = str(end_date)

    return config

def resolve_paths(config):
    paths_cfg = config['data']

    paths_cfg['raw_data_path'] = str(Path(paths_cfg['raw_data_path']) / config['run_id'])
    paths_cfg['processed_data_path'] = str(Path(paths_cfg['processed_data_path']) / config['run_id'])
    paths_cfg['split_data_path'] = str(Path(paths_cfg['split_data_path']) / config['run_id'])

    return config

def resolve_hyperparams(config):
    if 'model' in config and 'hyperparams' in config['model']:
        params = config['model']['hyperparams']
        for key, value in params.items():
            if value == "None":
                params[key] = None
    return config