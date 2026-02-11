import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats as sp_stats


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
                actual = self.split_results[metric].values
                shuffle = self.shuffle_results[metric].values

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
        splits = self.split_results['split'].values
        rank_ic = self.split_results['rank_ic'].values

        slope, intercept, r_value, p_value,std_err = sp_stats.linregress(splits, rank_ic)

        mid_point = len(splits) // 2
        first_half_mean = np.mean(rank_ic[:mid_point])
        second_half_mean = np.mean(rank_ic[mid_point:])

        autocorr = np.corrcoef(rank_ic[:-1], rank_ic[1:])[0,1]
        return {
            'trend_slope': slope,
            'trend_pvalue': p_value,
            'trend_r_squared': r_value ** 2,
            'first_half_mean_ic': first_half_mean,
            'second_half_mean_ic': second_half_mean,
            'autocorrelation': autocorr
        }

    def generate_summary_report(self) -> Dict:
        report = {
            'stability': self.analyse_stability('rank_ic'),
            'temporal': self.temporal_analysis(),
            'metric_correlations': self.metric_correlation_analysis().to_dict(),
            'outlier_splits': self.identify_outlier_splits('rank_ic')
        }

        if self.shuffle_results is not None:
            report['actual_vs_shuffle'] = self.compare_actual_v_shuffle()

        return report

    def print_diagnostic_report(self):
        print("=" * 70)
        print("MODEL DIAGNOSTICS REPORT")
        print("=" * 70)

        stability = self.analyse_stability('rank_ic')

        print(f"\n📊 PERFORMANCE STABILITY")
        print(f"   Coefficient of Variation: {stability['coefficient_of_variation']:.4f}")
        print(f"   Interquartile Range:      {stability['interquartile_range']:.4f}")
        print(f"   Stable Splits:            {stability['stable_splits']}/{len(self.split_results)}")
        print(f"   Low Outliers:             {stability['outliers_low']}")
        print(f"   High Outliers:            {stability['outliers_high']}")

        print(f"\n🎯 PERFORMANCE DISTRIBUTION")
        for category, count in stability['category_distribution'].items():
            print(f"   {category:12s}: {count:2d} splits")

        print(f"\n📈 TEMPORAL ANALYSIS")
        temporal = self.temporal_analysis()
        print(f"   Trend Slope:         {temporal['trend_slope']:.6f}")
        print(f"   Trend P-value:       {temporal['trend_pvalue']:.4f}")

        if temporal['trend_pvalue'] < 0.05:
            trend_dir = "📈 IMPROVING" if temporal['trend_slope'] > 0 else "📉 DEGRADING"
            print(f"   {trend_dir} over time (p < 0.05)")
        else:
            print(f"   ➡️  STABLE over time")

        print(f"   First Half IC:       {temporal['first_half_mean_ic']:.4f}")
        print(f"   Second Half IC:      {temporal['second_half_mean_ic']:.4f}")
        print(f"   Autocorrelation:     {temporal['autocorrelation']:.4f}")

        if self.shuffle_results is not None:
            print(f"\n🎲 ACTUAL VS SHUFFLE")
            comp = self.compare_actual_v_shuffle()

            for metric, values in comp.items():
                print(f"\n   {metric.upper()}:")
                print(f"      Actual:  {values['actual_mean']:.4f} ± {values['actual_std']:.4f}")
                print(f"      Shuffle: {values['shuffle_mean']:.4f} ± {values['shuffle_std']:.4f}")
                print(f"      Δ:       {values['difference']:.4f} (Effect size: {values['effect_size']:.2f})")

        outliers = self.identify_outlier_splits()
        if outliers:
            print(f"\n⚠️  OUTLIER SPLITS (>2σ from mean): {outliers}")

        print("=" * 70)

class RegimeAnalyser:
    def __init__(self, splits, splits_results):
        self.splits = splits
        self.splits_results = splits_results

    def extract_regime_features(self):
        regime_data = []

        for idx, ((train, test), result) in enumerate(zip(self.splits, self.splits_results.to_dict('records'))):
            test_dates = test.index.get_level_values('Date')