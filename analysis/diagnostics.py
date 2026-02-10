import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RegimeAnalysis:
    regime_df: pd.DataFrame
    correlations: pd.Series
    good_regime_characteristics: Dict
    bad_regime_characteristics: Dict
    regime_clusters: Optional[pd.DataFrame] = None


class ModelDiagnostics:
    def __init__(self, split_results, shuffle_results):
        self.split_results = pd.DataFrame(split_results)
        self.shuffle_results = pd.DataFrame(shuffle_results)

    def analyse_stability(self, metric = 'rank_ic'):
        print(self.split_results)
        values = self.split_results[metric].values

        coefvariation = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        q3, q1 = np.percentile(values, [0.75, 0.25])
        interquartile = q3 - q1

        stable_splits = np.sum((values >= q1) & (values <= q3))
        outlier_splits_low = np.sum(values < (q1 - 1.5 * interquartile))
        outlier_splits_high = np.sum(values > (q3 + 1.5 * interquartile))

        categories = pd.cut(values, bins=[-np.inf, 0, 0,2, 0,4, np.inf],labels=['Negative', 'Weak', 'Moderate', 'Strong'])

        print(coefvariation, q3, q1, interquartile, stable_splits, outlier_splits_low, outlier_splits_high, categories)
        return {
            'coefficient_of_variation': coefvariation,
            'interquartile_range': interquartile,
            'stable_splits': stable_splits,
            'outliers_low': outlier_splits_low,
            'outliers_high': outlier_splits_high,
            'category_distribution': categories.value_counts().to_dict(),
            'worst_split': {
                'index': int(values.argmin()),
                'value': float(values.min())
            },
            'best_split': {
                'index': int(values.argmax()),
                'value': float(values.max())
            }
        }