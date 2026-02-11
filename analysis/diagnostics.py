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
        values = self.split_results[metric].values

        coefvariation = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        q3, q1 = np.percentile(values, [0.75, 0.25])
        interquartile = q3 - q1

        stable_splits = np.sum((values >= q1) & (values <= q3))
        outlier_splits_low = np.sum(values < (q1 - 1.5 * interquartile))
        outlier_splits_high = np.sum(values > (q3 + 1.5 * interquartile))

        categories = pd.cut(values, bins=[-np.inf, 0, 0.2, 0.4, np.inf],labels=['Negative', 'Weak', 'Moderate', 'Strong'])

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

    def compare_actual_v_shuffle(self):
        if self.shuffle_results is None:
            raise ValueError("Shuffle Results Not Available")

        comparison = {}

        for metric in ['mse', 'r2', 'rank_ic']:
            if metric in self.split_results.columns and metric in self.shuffle_results.columns:
                actual = self.split_results['metric'].values
                shuffle = self.shuffle_results['metric'].values

                comparison[metric] = {
                    'actual_mean': float(np.mean(actual)),
                    'shuffle_mean': float(np.mean(shuffle)),
                    'actual_std': float(np.std(actual)),
                    'shuffle_std': float(np.std(shuffle)),
                    'difference': float(np.mean(actual) - np.mean(shuffle)),
                    'difference_std': float(np.std(actual) - np.std(shuffle)),
                    'effect_size': float((np.mean(actual) - np.mean(shuffle)) / np.std(shuffle)) if np.std(shuffle) > 0 else 0
                }
        print("comparison", comparison)
        return comparison

    def identify_outlier_splits(self, metric = 'rank_ic', threshold = 2.0):
        values = self.split_results[metric].values
        mean = np.mean(values)
        std = np.std(values)

        z_scores = np.abs((values - mean) / std)
        outliers = np.where(z_scores > threshold)[0]
        print("Outliers:", outliers)
        return outliers


    def metric_correlation_analysis(self):
        metric_cols = [col for col in self.split_results.columns if col != 'split']
        print("correlation:",self.split_results[metric_cols].corr() )
        return self.split_results[metric_cols].corr()

    def temporal_analysis(self):
        pass

    def generate_summary_report(self):
        pass

    def print_diagnostic_report(self):
        pass