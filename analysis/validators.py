import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ValidationResults:
    ic_mean: float
    ic_std: float
    ic_median: float
    information_ratio: float
    t_statistic: float
    p_value: float
    confidence_interval_95: Tuple[float, float]
    positive_split_ratio: float
    vs_shuffle_t_stat: Optional[float] = None
    vs_shuffle_p_value: Optional[float] = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def beats_random(self, alpha: float = 0.05) -> bool:
        if self.vs_shuffle_p_value is None:
            return False
        return self.vs_shuffle_p_value < alpha

    def to_dict(self) -> Dict:
        return {
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_median': self.ic_median,
            'information_ratio': self.information_ratio,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'ci_lower': self.confidence_interval_95[0],
            'ci_upper': self.confidence_interval_95[1],
            'positive_split_ratio': self.positive_split_ratio,
            'vs_shuffle_t_stat': self.vs_shuffle_t_stat,
            'vs_shuffle_p_value': self.vs_shuffle_p_value,
            'is_significant': self.is_significant(),
            'beats_random': self.beats_random()
        }


class StatisticalValidator:

    def __init__(self, split_results: List[Dict], shuffle_results: Optional[List[Dict]] = None):
        self.split_results = pd.DataFrame(split_results)
        self.shuffle_results = pd.DataFrame(shuffle_results) if shuffle_results else None

    def validate_rank_ic(self, alpha: float = 0.05) -> ValidationResults:
        ics = self.split_results['rank_ic'].values

        ic_mean = np.mean(ics)
        ic_std = np.std(ics, ddof=1)
        ic_median = np.median(ics)

        information_ratio = ic_mean / ic_std if ic_std > 0 else 0

        t_stat, p_value = stats.ttest_1samp(ics, 0)

        ci_95 = stats.t.interval(
            0.95,
            len(ics) - 1,
            loc=ic_mean,
            scale=stats.sem(ics)
        )

        positive_ratio = np.sum(ics > 0) / len(ics)

        vs_shuffle_t = None
        vs_shuffle_p = None
        if self.shuffle_results is not None:
            shuffle_ics = self.shuffle_results['rank_ic'].values
            vs_shuffle_t, vs_shuffle_p = stats.ttest_ind(ics, shuffle_ics)

        return ValidationResults(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_median=ic_median,
            information_ratio=information_ratio,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval_95=ci_95,
            positive_split_ratio=positive_ratio,
            vs_shuffle_t_stat=vs_shuffle_t,
            vs_shuffle_p_value=vs_shuffle_p
        )

    def validate_all_metrics(self) -> Dict[str, ValidationResults]:
        results = {}

        for metric in ['rank_ic', 'r2', 'mse']:
            if metric in self.split_results.columns:
                if metric == 'mse':
                    results[metric] = self._validate_metric_lower(metric)
                else:
                    results[metric] = self._validate_metric_higher(metric)

        return results

    def _validate_metric_higher(self, metric: str) -> ValidationResults:
        values = self.split_results[metric].values

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        median_val = np.median(values)

        info_ratio = mean_val / std_val if std_val > 0 else 0
        t_stat, p_value = stats.ttest_1samp(values, 0)

        ci_95 = stats.t.interval(
            0.95, len(values) - 1,
            loc=mean_val, scale=stats.sem(values)
        )

        positive_ratio = np.sum(values > 0) / len(values)

        vs_shuffle_t = None
        vs_shuffle_p = None
        if self.shuffle_results is not None and metric in self.shuffle_results.columns:
            shuffle_vals = self.shuffle_results[metric].values
            vs_shuffle_t, vs_shuffle_p = stats.ttest_ind(values, shuffle_vals)

        return ValidationResults(
            ic_mean=mean_val,
            ic_std=std_val,
            ic_median=median_val,
            information_ratio=info_ratio,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval_95=ci_95,
            positive_split_ratio=positive_ratio,
            vs_shuffle_t_stat=vs_shuffle_t,
            vs_shuffle_p_value=vs_shuffle_p
        )

    def _validate_metric_lower(self, metric: str) -> ValidationResults:
        values = self.split_results[metric].values

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        median_val = np.median(values)

        info_ratio = -mean_val / std_val if std_val > 0 else 0

        if self.shuffle_results is not None and metric in self.shuffle_results.columns:
            shuffle_vals = self.shuffle_results[metric].values
            baseline = np.mean(shuffle_vals)
        else:
            baseline = mean_val

        t_stat, p_value = stats.ttest_1samp(values, baseline)

        ci_95 = stats.t.interval(
            0.95, len(values) - 1,
            loc=mean_val, scale=stats.sem(values)
        )

        positive_ratio = np.sum(values < baseline) / len(values)

        vs_shuffle_t = None
        vs_shuffle_p = None
        if self.shuffle_results is not None and metric in self.shuffle_results.columns:
            shuffle_vals = self.shuffle_results[metric].values
            vs_shuffle_t, vs_shuffle_p = stats.ttest_ind(values, shuffle_vals)

        return ValidationResults(
            ic_mean=mean_val,
            ic_std=std_val,
            ic_median=median_val,
            information_ratio=info_ratio,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval_95=ci_95,
            positive_split_ratio=positive_ratio,
            vs_shuffle_t_stat=vs_shuffle_t,
            vs_shuffle_p_value=vs_shuffle_p
        )

    def print_validation_report(self, metric: str = 'rank_ic'):
        result = self.validate_rank_ic() if metric == 'rank_ic' else self._validate_metric_higher(metric)

        print("=" * 70)
        print(f"STATISTICAL VALIDATION REPORT - {metric.upper()}")
        print("=" * 70)

        print(f"\n📊 DESCRIPTIVE STATISTICS")
        print(f"   Mean:              {result.ic_mean:>8.4f}")
        print(f"   Std Dev:           {result.ic_std:>8.4f}")
        print(f"   Median:            {result.ic_median:>8.4f}")
        print(f"   Information Ratio: {result.information_ratio:>8.4f}")

        print(f"\n🎯 HYPOTHESIS TEST (H0: {metric} = 0)")
        print(f"   T-statistic:       {result.t_statistic:>8.4f}")
        print(f"   P-value:           {result.p_value:>8.4f}")
        print(f"   95% CI:            [{result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}]")

        if result.is_significant():
            print(f"   ✅ SIGNIFICANT at α=0.05")
        else:
            print(f"   ❌ NOT significant at α=0.05")

        print(f"\n📈 CONSISTENCY")
        print(f"   Positive splits:   {result.positive_split_ratio * 100:>6.1f}%")

        if self.shuffle_results is not None:
            print(f"\n🎲 COMPARISON VS SHUFFLE TEST")
            print(f"   T-statistic:       {result.vs_shuffle_t_stat:>8.4f}")
            print(f"   P-value:           {result.vs_shuffle_p_value:>8.4f}")

            if result.beats_random():
                print(f"   ✅ BEATS RANDOM at α=0.05")
            else:
                print(f"   ❌ DOES NOT beat random at α=0.05")

        print(f"\n💡 INTERPRETATION")
        if result.is_significant() and result.beats_random():
            print("   ✅ Model demonstrates statistically significant predictive power")
            print("   ✅ Performance significantly better than random")
            if result.information_ratio > 1.0:
                print("   ✅ Strong information ratio (>1.0)")
            elif result.information_ratio > 0.5:
                print("   ℹ️  Moderate information ratio (0.5-1.0)")
            else:
                print("   ⚠️  Weak information ratio (<0.5) - high variance")
        elif result.is_significant():
            print("   ⚠️  Model shows signal but may not beat random baseline robustly")
        else:
            print("   ❌ No evidence of statistically significant predictive power")

        print("=" * 70)

    def export_results(self, filepath: str):

        results = self.validate_all_metrics()

        output = {
            metric: result.to_dict()
            for metric, result in results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results exported to {filepath}")