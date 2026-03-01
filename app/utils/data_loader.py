import json

import pandas as pd

from config import ARTIFACTS_DIR
import streamlit as st


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

def load_backtest_results(run_id):
    path = run_id / 'backtest' / 'metrics.json'
    st.write(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_backtest_returns(run_id):
    path = run_id / 'backtest' / 'returns_timeseries.csv'
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_all_experiment_metrics():
    rows = []
    for exp in list_experiment():
        for run in list_runs(exp):
            val = load_validation_results(run)
            bt = load_backtest_results(run)
            if val and bt:
                rows.append({
                    "experiment": exp,
                    "run": run,
                    "ic_mean": val.get("rank_ic").get("ic_mean"),
                    "bt_get": bt.get("Sharpe Ratio"),
                    "max_drawdown": bt.get("Max Drawdown")
                })
    return pd.DataFrame(rows)

