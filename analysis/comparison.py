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

        self.experiments_df = pd.DataFrame()
        self.metrics_df     = pd.DataFrame()
        self.summaries      = {}
        self.artifacts_df   = pd.DataFrame()
        self._build_frames()

    def _execute_in_clause(self, ids):
        placeholders = ",".join(["?"] * len(ids))
        return placeholders

    def get_experiment_data(self, ids):
        if not ids:
            return []

        placeholders = self._execute_in_clause(ids)
        query = f"""
            SELECT run_id, name, model_type, dataset_path, config_json, created_at
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

    def _build_frames(self):
        if self.experiment_data:
            self.experiments_df = pd.DataFrame(
                self.experiment_data,
                columns=["run_id","name","model_type","dataset_path", "config_json","created_at"]
            )
        if self.metrics_data:
            self.metrics_df = pd.DataFrame(
                self.metrics_data,
                columns=["experiment_id", "split", "mse", "r2", "rank_ic"]
            )
            if not self.experiments_df.empty:
                nm = self.experiments_df.set_index("run_id")["name"].to_dict()
                self.metrics_df["model_name"] = self.metrics_df["experiment_id"].map(nm)
                
        for eid, js in self.summary_data:
            try:    self.summaries[eid] = json.loads(js)
            except: self.summaries[eid] = {}
            
        if self.artifacts_data:
            self.artifacts_df = pd.DataFrame(
                self.artifacts_data,
                columns=["experiment_id","model_path","predictions_path",
                         "plots_path","feature_importance_path"])

