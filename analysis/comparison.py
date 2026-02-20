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
            SELECT experiment_id, model_path, predictions_path, plots_path, analysis_path
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
                         "plots_path","analysis_path"])

    def ic_statistics(self):
        if self.metrics_df.empty:
            return pd.DataFrame()

        records = []
        print(self.metrics_df)
        for eid, grp in self.metrics_df.groupby("experiment_id"):
            ic = grp["rank_ic"].dropna()
            if len(ic) < 2: continue
            ir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
            records.append({
                "experiment_id": eid,
                "model_name": grp["model_name"].iloc[0],
                "ic_mean": round(ic.mean(), 4),
                "ic_std": round(ic.std(), 4),
                "ic_ir": round(ir, 4),
                "hit_rate": round((ic > 0).mean(), 4),
                "ic_min": round(ic.min(), 4),
                "ic_max": round(ic.max(), 4),
                "n_splits": len(ic),
            })
        return pd.DataFrame(records)

    def retrieve_analysis_data(self, file, type):
        if self.artifacts_df.empty:
            return None

        results = {}

        for _, row in self.artifacts_df.iterrows():
            eid = row["experiment_id"]
            analysis_path = Path(row["analysis_path"])
            val_file = analysis_path / file

            if not val_file.exists():
                print(f"[{eid}] File not found:", val_file)
                continue

            if type == "json":
                with open(val_file, "r") as f:
                    data = json.load(f)
            else:
                data = pd.read_parquet(val_file)

            results[eid] = data

        if not results:
            return None

        if type != "json":
            combined = []
            for eid, df in results.items():
                df = df.copy()
                df["experiment_id"] = eid
                combined.append(df)
            return pd.concat(combined, ignore_index=True)

        return results
    def get_validation_data(self):
        data = self.retrieve_analysis_data('validation_results.json', 'json')
        return data

    def get_feature_analysis(self):
        data = self.retrieve_analysis_data('feature_analysis.json', 'json')
        return data

    def get_diagnostics_analysis(self):
        data = self.retrieve_analysis_data('diagnostics_summary.json', 'json')
        return data

    def get_feature_importance(self):
        data = self.retrieve_analysis_data('feature_importance.parquet', 'parquet')
        return data

    def get_regime_analysis(self):
        data = self.retrieve_analysis_data('regime_analysis.parquet', 'parquet')
        data = data.set_index(['experiment_id','split'])
        return data



model = CompareModels(experiment_ids=['20260219_210026','20260220_214631'])
print(model.get_regime_analysis())
