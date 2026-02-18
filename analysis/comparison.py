import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class CompareModels:
    def __init__(self, experiment_ids=[], db_path='../data/experiments/tracking.db'):
        self.experiment_ids = experiment_ids
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.experiment_data = self.get_experiment_data(experiment_ids)
        self.metrics_data = self.get_metrics_data(experiment_ids)
        self.summary_data = self.get_summary_data(experiment_ids)
        self.artifacts_data = self.get_artifacts(experiment_ids)

    def _execute_in_clause(self, ids):
        placeholders = ",".join(["?"] * len(ids))
        return placeholders

    def get_experiment_data(self, ids):
        if not ids:
            return []

        placeholders = self._execute_in_clause(ids)
        query = f"""
            SELECT run_id, name, model_type, dataset_path, created_at
            FROM experiments
            WHERE run_id IN ({placeholders})
            ORDER BY created_at DESC
            """
        self.cursor.execute(query, ids)
        return self.cursor.fetchall()

    def get_metrics_data(self, ids):
        if not ids:
            return []
        placeholders = self._execute_in_clause(ids)
        query = f"""
            SELECT experiment_id, split, mse, r2, rank_ic
            FROM metrics
            WHERE experiment_id IN ({placeholders})
            ORDER BY experiment_id, split
            """
        self.cursor.execute(query, ids)
        return self.cursor.fetchall()

    def get_summary_data(self, ids):
        if not ids:
            return []
        placeholders = self._execute_in_clause(ids)
        query = f"""
            SELECT experiment_id, summary_json
            FROM summaries
            WHERE experiment_id IN ({placeholders})
            """
        self.cursor.execute(query, ids)
        return self.cursor.fetchall()

    def get_artifacts(self, ids):
        if not ids:
            return []
        placeholders = self._execute_in_clause(ids)
        query = f"""
            SELECT experiment_id, model_path, predictions_path, plots_path, feature_importance_path
            FROM artifacts
            WHERE experiment_id IN ({placeholders})
            """
        self.cursor.execute(query, ids)
        return self.cursor.fetchall()

