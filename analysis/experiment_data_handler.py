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

class ExperimentDataRepo:
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
        if not data:
            return None

        records = []

        for eid, content in data.items():
            row = {"experiment_id": eid}

            for metric_name, stats_dict in content.items():
                row[f"{metric_name}_mean"] = stats_dict.get("ic_mean")
                row[f"{metric_name}_std"] = stats_dict.get("ic_std")
                row[f"{metric_name}_median"] = stats_dict.get("ic_median")
                row[f"{metric_name}_ir"] = stats_dict.get("information_ratio")
                row[f"{metric_name}_t_stat"] = stats_dict.get("t_statistic")
                row[f"{metric_name}_p_value"] = stats_dict.get("p_value")
                row[f"{metric_name}_ci_lower"] = stats_dict.get("ci_lower")
                row[f"{metric_name}_ci_upper"] = stats_dict.get("ci_upper")
                row[f"{metric_name}_positive_ratio"] = stats_dict.get("positive_split_ratio")
                row[f"{metric_name}_vs_shuffle_t"] = stats_dict.get("vs_shuffle_t_stat")
                row[f"{metric_name}_vs_shuffle_p"] = stats_dict.get("vs_shuffle_p_value")
                row[f"{metric_name}_significant"] = stats_dict.get("is_significant")
                row[f"{metric_name}_beats_random"] = stats_dict.get("beats_random")

            records.append(row)

        df = pd.DataFrame(records)

        if not self.experiments_df.empty:
            name_map = self.experiments_df.set_index("run_id")["name"]
            df["model_name"] = df["experiment_id"].map(name_map)

        return df.set_index("experiment_id")

    def get_feature_analysis(self):
        data = self.retrieve_analysis_data('feature_analysis.json', 'json')
        if not data:
            return None

        records = []

        for eid, content in data.items():
            row = {"experiment_id": eid}

            group_stats = content.get("group_stats", {})

            for group_name, stats_dict in group_stats.items():
                row[f"{group_name}_total_imp"] = stats_dict.get("total_importance")
                row[f"{group_name}_mean_imp"] = stats_dict.get("mean_importance")
                row[f"{group_name}_top_imp"] = stats_dict.get("top_importance")

            reduction = content.get("feature_reduction", {})
            row["n_features_80pct"] = len(reduction.get("80pct", []))
            row["n_features_90pct"] = len(reduction.get("90pct", []))
            row["n_features_95pct"] = len(reduction.get("95pct", []))

            redundant_pairs = content.get("redundant_pairs", [])
            row["n_redundant_pairs"] = len(redundant_pairs)

            if redundant_pairs:
                correlations = [p["correlation"] for p in redundant_pairs]
                row["max_redundancy_corr"] = max(correlations)
                row["mean_redundancy_corr"] = np.mean(correlations)
            else:
                row["max_redundancy_corr"] = None
                row["mean_redundancy_corr"] = None

            records.append(row)

        df = pd.DataFrame(records)

        if not self.experiments_df.empty:
            name_map = self.experiments_df.set_index("run_id")["name"]
            df["model_name"] = df["experiment_id"].map(name_map)

        return df.set_index("experiment_id")

    def get_diagnostics_analysis(self):
        data = self.retrieve_analysis_data('diagnostics_summary.json', 'json')
        if not data:
            return None

        records = []

        for eid, content in data.items():
            row = {"experiment_id": eid}

            stability = content.get("stability", {})
            row.update({
                "cv": stability.get("coefficient_of_variation"),
                "iqr": stability.get("interquartile_range"),
                "stable_splits": stability.get("stable_splits"),
                "outliers_low": stability.get("outliers_low"),
                "outliers_high": stability.get("outliers_high"),
                "best_ic": stability.get("best_split", {}).get("value"),
                "worst_ic": stability.get("worst_split", {}).get("value"),
            })

            temporal = content.get("temporal", {})
            row.update({
                "trend_slope": temporal.get("trend_slope"),
                "trend_pvalue": temporal.get("trend_pvalue"),
                "trend_r2": temporal.get("trend_r_squared"),
                "first_half_ic": temporal.get("first_half_mean_ic"),
                "second_half_ic": temporal.get("second_half_mean_ic"),
                "autocorr": temporal.get("autocorrelation"),
            })

            avs = content.get("actual_vs_shuffle", {}).get("rank_ic", {})
            row.update({
                "ic_actual_mean": avs.get("actual_mean"),
                "ic_shuffle_mean": avs.get("shuffle_mean"),
                "ic_effect_size": avs.get("effect_size"),
            })

            records.append(row)

        df = pd.DataFrame(records)

        if not self.experiments_df.empty:
            name_map = self.experiments_df.set_index("run_id")["name"]
            df["model_name"] = df["experiment_id"].map(name_map)

        return df.set_index("experiment_id")

    def get_feature_importance(self):
        data = self.retrieve_analysis_data('feature_importance.parquet', 'parquet')
        return data

    def get_regime_analysis(self):
        data = self.retrieve_analysis_data('regime_analysis.parquet', 'parquet')
        data = data.set_index(['experiment_id','split'])
        return data

