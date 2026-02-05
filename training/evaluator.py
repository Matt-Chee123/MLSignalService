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
            values = np.array(
                [r[metric_name] for r in split_results if not np.isnan(r[metric_name])]
            )

            aggregated[f"{metric_name}_mean"] = values.mean() if len(values) else np.nan
            aggregated[f"{metric_name}_std"] = values.std() if len(values) else np.nan

        return aggregated
