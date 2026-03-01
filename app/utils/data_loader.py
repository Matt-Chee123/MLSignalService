import json
from app.config import ARTIFACTS_DIR
from pathlib import Path


def list_experiment():
    directories = []
    for dir in ARTIFACTS_DIR.iterdir():
        if dir.is_dir():
            directories.append(dir)

    return directories

def list_runs(experiment):
    runs = []
    run_dir = ARTIFACTS_DIR / experiment
    for run in sorted(run_dir.iterdir()):
        if run.is_dir():
            runs.append(run)

    return runs

def load_validation_results(experiment, run_id):
    path = ARTIFACTS_DIR / experiment / run_id / 'metrics'


experiments = list_experiment()

for experiment in experiments:
    list_runs(experiment)