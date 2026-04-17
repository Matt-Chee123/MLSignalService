import json

import pandas as pd
from pathlib import Path
import streamlit as st

ARTIFACTS_DIR = Path('../training/artifacts')

def list_experiment():
    directories = []
    for dir in ARTIFACTS_DIR.iterdir():
        if dir.is_dir():
            directories.append(dir)

    return directories

def list_runs(experiment):
    runs = []
    for run in sorted(experiment.iterdir()):
        if run.is_dir():
            runs.append(run)

    return runs

def load_validation_results(run_id):
    path = run_id / 'analysis' / 'validation_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_diagnostics_results(run_id):
    path = run_id / 'analysis' / 'diagnostics_summary.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_regime_analysis(run_id):
    path = run_id / 'analysis' / 'regime_analysis.parquet'
    if path.exists():
        return pd.read_parquet(path)
    return None

def load_backtest_results(run_id):
    path = run_id / 'backtest' / 'metrics.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_backtest_returns(run_id):
    path = run_id / 'backtest' / 'returns_timeseries.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def load_latest_predictions(run_id):
    path = run_id / 'predictions' / 'live_signal.csv'

    if path.exists():
        df = pd.read_csv(path)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        return df

    return None


@st.cache_data
def load_all_experiment_metrics():

    rows = []

    for exp in list_experiment():
        for run in list_runs(exp):
            val = load_validation_results(run)
            bt = load_backtest_results(run)

            if val and bt:
                rank_ic = val.get('rank_ic', {})
                ic_mean = rank_ic.get('ic_mean') if isinstance(rank_ic, dict) else None

                rows.append({
                    "experiment": exp,
                    "run": run,
                    "ic_mean": ic_mean,
                    "sharpe_net": bt.get("Sharpe Ratio"),
                    "max_drawdown": bt.get("Max Drawdown")
                })

    return pd.DataFrame(rows)

def load_config(run_id):
    path = run_id / 'config.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_run_metadata(run_id):

    config = load_config(run_id)

    metadata = {
        'experiment_name': run_id.parent.name,
        'run_id': run_id.name,
        'date': run_id.name.split('_')[0] if '_' in run_id.name else 'Unknown'
    }

    if config:
        metadata['model_type'] = config.get('model', {}).get('model_type', 'Unknown')
        metadata['n_features'] = 'Unknown'
        metadata['horizon'] = config.get('horizon', 0)

    return metadata

def load_feature_importance(run_id):
    path = run_id / "analysis" / "feature_importance.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

def load_feature_analysis(run_id):
    path = run_id / "analysis" / "feature_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
