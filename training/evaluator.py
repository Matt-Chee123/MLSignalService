import numpy as np
from typing import Callable, Dict, List, Any


class Evaluator:
    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics

    def evaluate_split(
        self,
        y_true,
        y_pred,
        *,
        signal=None,
        future_returns=None
    ) -> Dict[str, float]:
        results = {}

        for name, metric_fn in self.metrics.items():
            try:
                if name in ["rank_ic"]:
                    results[name] = metric_fn(signal, future_returns)

                elif name in ["prediction_autocorr"]:
                    results[name] = metric_fn(signal)

                else:
                    results[name] = metric_fn(y_true, y_pred)

            except Exception:
                results[name] = np.nan

        return results

    def evaluate_all_splits(
        self,
        split_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        aggregated = {}

        for metric_name in split_results[0].keys():
            if metric_name == 'split':
                continue

            values = np.array(
                [r[metric_name] for r in split_results if not np.isnan(r[metric_name])]
            )

            aggregated[f"{metric_name}_mean"] = values.mean() if len(values) else np.nan
            aggregated[f"{metric_name}_std"] = values.std() if len(values) else np.nan

        return aggregated

    def _safe_get(self, d, key, default=None):
        return d.get(key, default) if d else default

    def _t_stat(self, mean, std, n):
        if mean is None or std in (None, 0) or n <= 1:
            return None
        return mean / (std / np.sqrt(n))

    def pass_validation(self, split_metrics, shuffle_metrics):

        if not split_metrics:
            return {"pass": False, "reason": "Missing split metrics"}

        n = self._safe_get(split_metrics, "num_splits", 1)

        rank_ic_mean = self._safe_get(split_metrics, "rank_ic_mean")
        rank_ic_std = self._safe_get(split_metrics, "rank_ic_std")

        hit_rate_mean = self._safe_get(split_metrics, "hit_rate_mean")

        sharpe_mean = self._safe_get(split_metrics, "sharpe_mean")
        sharpe_std = self._safe_get(split_metrics, "sharpe_std")

        max_dd = self._safe_get(split_metrics, "max_drawdown_mean")

        shuffle_sharpe = self._safe_get(shuffle_metrics, "sharpe_mean")
        shuffle_rank_ic = self._safe_get(shuffle_metrics, "rank_ic_mean")

        rank_ic_t = self._t_stat(rank_ic_mean, rank_ic_std, n)
        sharpe_t = self._t_stat(sharpe_mean, sharpe_std, n)

        conditions = {}
        skipped = []

        if rank_ic_mean is not None:
            conditions["rank_ic_positive"] = rank_ic_mean > 0.02
        else:
            skipped.append("rank_ic_positive")

        if rank_ic_t is not None:
            conditions["rank_ic_significant"] = rank_ic_t > 2.0
        else:
            skipped.append("rank_ic_significant")

        if shuffle_rank_ic is not None and rank_ic_mean is not None:
            conditions["rank_ic_beats_shuffle"] = rank_ic_mean > shuffle_rank_ic
        else:
            skipped.append("rank_ic_beats_shuffle")

        if hit_rate_mean is not None:
            conditions["hit_rate_above_random"] = hit_rate_mean > 0.52
        else:
            skipped.append("hit_rate_above_random")

        if sharpe_mean is not None:
            conditions["sharpe_positive"] = sharpe_mean > 0.5
        else:
            skipped.append("sharpe_positive")

        if sharpe_mean is not None and shuffle_sharpe is not None:
            conditions["sharpe_beats_shuffle"] = sharpe_mean > shuffle_sharpe
        else:
            skipped.append("sharpe_beats_shuffle")

        if max_dd is not None:
            conditions["drawdown_controlled"] = max_dd < 0.30
        else:
            skipped.append("drawdown_controlled")

        evaluated_conditions = [v for v in conditions.values() if v is not None]

        passed = (
                len(evaluated_conditions) > 0
                and all(evaluated_conditions)
        )

        return {
            "pass": passed,
            "conditions_checked": conditions,
            "conditions_skipped": skipped,
            "statistics": {
                "rank_ic_t": rank_ic_t,
                "sharpe_t": sharpe_t,
            },
        }
