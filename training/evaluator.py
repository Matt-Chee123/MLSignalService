from training import metrics
import numpy as np

class Evaluator:
    def __init__(self, metrics_list = None):
        self.metrics_list = metrics_list or []

    def evaluate_split(self, y_true, y_preds, extra=None):
        results = {}
        for metric_fn in self.metrics_list:
            results[metric_fn.__name__] = metric_fn(y_true, y_preds, extra)

        return results

    def evaluate_all_splits(self, splits):
        all_splits = []
        for split in splits:
            all_splits.append(self.evaluate_split(*split))

        agg_results = {}
        for key in all_splits[0].keys():
            values = [r[key] for r in all_splits]
            agg_results[f'{key}_mean'] = np.mean(values)
            agg_results[f'{key}_std'] = np.std(values)
        return agg_results