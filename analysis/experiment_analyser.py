import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from analysis.experiment_data_handler import ExperimentDataRepo
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    def __init__(self, metrics_df=None, validation_df=None, diagnostics_df=None, feature_df=None, regime_df=None):
        self.metrics_df = metrics_df
        self.validation_df = validation_df
        self.diagnostics_df = diagnostics_df
        self.feature_df = feature_df
        self.regime_df = regime_df

    def compare_ic(self, annualize=False, sort_by="ic_ir"):
        if self.metrics_df is None or self.metrics_df.empty:
            return None

        records = []

        for eid, grp in self.metrics_df.exp_groupsby("experiment_id"):
            ic = grp["rank_ic"].dropna()

            if len(ic) < 2:
                continue

            mean_ic = ic.mean()
            std_ic = ic.std(ddof=1)

            ir = mean_ic / std_ic if std_ic > 0 else np.nan
            hit_rate = (ic > 0).mean()

            t_stat, p_value = stats.ttest_1samp(ic, 0)

            if annualize:
                ir *= np.sqrt(len(ic))

            records.append({
                "experiment_id": eid,
                "model_name": grp.get("model_name", pd.Series([None])).iloc[0],
                "ic_mean": round(mean_ic, 4),
                "ic_std": round(std_ic, 4),
                "ic_ir": round(ir, 4),
                "hit_rate": round(hit_rate, 4),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "n_splits": len(ic),
            })

        df = pd.DataFrame(records)

        if df.empty:
            return None

        return df.sort_values(sort_by, ascending=False).set_index("experiment_id")

    def compare_stability(self):
        if self.diagnostics_df is None or self.diagnostics_df.empty:
            return []

        records = []
        for experiment_id, data in self.diagnostics_df.iterrows():
            coefficient_variation = data['cv'] + data['ic_actual_mean']

            records.append({
                'id': experiment_id,
                'coefficient_variation': coefficient_variation,
                'stable_splits': data['stable_splits'],
                'outliers_high': data['outliers_high'],
                'outliers_low': data['outliers_low'],
                'trend_slope': data['trend_slope'],
                'decay_rate': data['second_half_ic'] / data['first_half_ic'],
                'autocorrelation': data['autocorr'],
                'effect_size': data['ic_effect_size'],
            })
        return pd.DataFrame(records)

    def compare_regime_robustness(self):
        if self.regime_df is None or self.regime_df.empty:
            return pd.DataFrame()

        results = []

        for exp_id, group in self.regime_df.groupby(level="experiment_id"):
            mean_ic = group["rank_ic"].mean()
            std_ic = group["rank_ic"].std()
            cv_ic = std_ic / abs(mean_ic) if mean_ic != 0 else None
            worst_ic = group["rank_ic"].min()
            skew_ic = group["rank_ic"].skew()
            positive_split_ratio = (group["rank_ic"] > 0).mean()

            down_mask = group["market_return"] < 0
            down_ic = group.loc[down_mask, "rank_ic"].mean()

            stress_threshold = group["max_drawdown"].quantile(0.8)
            stress_mask = group["max_drawdown"] <= stress_threshold
            stress_ic = group.loc[stress_mask, "rank_ic"].mean()

            vol_corr = group["rank_ic"].corr(group["market_volatility"])
            dir_corr = group["rank_ic"].corr(group["market_return"])

            results.append({
                "experiment": exp_id,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "cv_ic": cv_ic,
                "worst_ic": worst_ic,
                "down_ic": down_ic,
                "stress_ic": stress_ic,
                "vol_corr": vol_corr,
                "dir_corr": dir_corr,
                "skew_ic": skew_ic,
                "positive_split_ratio": positive_split_ratio
            })

        return pd.DataFrame(results)

    def compare_feature_risk(self):
        if self.feature_df is None or self.feature_df.empty:
            return pd.DataFrame()

        results = []

        for exp_id, row in self.feature_df.iterrows():
            total_cols = [c for c in row.index if c.endswith("_total_imp")]
            importances = row[total_cols].dropna()

            total_importance = importances.sum()

            weights = importances / total_importance if total_importance != 0 else importances

            top_weight = weights.max()
            top3_weight = weights.sort_values(ascending=False).head(3).sum()
            hhi = (weights ** 2).sum()

            redundancy_pairs = row["n_redundant_pairs"]
            max_corr = row["max_redundancy_corr"]
            mean_corr = row["mean_redundancy_corr"]

            n80 = row["n_features_80pct"]
            n90 = row["n_features_90pct"]
            n95 = row["n_features_95pct"]

            results.append({
                "experiment": exp_id,
                "top_feature_weight": top_weight,
                "top3_weight": top3_weight,
                "hhi_concentration": hhi,
                "n_features_80pct": n80,
                "n_features_90pct": n90,
                "n_features_95pct": n95,
                "n_redundant_pairs": redundancy_pairs,
                "max_redundancy_corr": max_corr,
                "mean_redundancy_corr": mean_corr
            })

        return pd.DataFrame(results)

repo = ExperimentDataRepo(experiment_ids=['20260219_210026','20260220_214631'])
analyzer = ModelAnalyzer(
    metrics_df=repo.metrics_df,
    validation_df=repo.get_validation_data(),
    diagnostics_df=repo.get_diagnostics_analysis(),
    feature_df=repo.get_feature_analysis(),
    regime_df=repo.get_regime_analysis(),
)
print(analyzer.compare_feature_risk())