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

        for eid, grp in self.metrics_df.groupby("experiment_id"):
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
        pass

    def compare_regime_robustness(self):
        pass

    def compare_feature_risk(self):
        pass

repo = ExperimentDataRepo(experiment_ids=['20260219_210026','20260220_214631'])
analyzer = ModelAnalyzer(
    metrics_df=repo.metrics_df,
    validation_df=repo.get_validation_data(),
    diagnostics_df=repo.get_diagnostics_analysis(),
    feature_df=repo.get_feature_analysis(),
    regime_df=repo.get_regime_analysis(),
)
ic_comparison = analyzer.compare_ic()
print(ic_comparison)