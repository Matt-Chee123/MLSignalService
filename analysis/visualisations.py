
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy import stats as sp_stats

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ValidationVisualizer:

    @staticmethod
    def plot_actual_vs_shuffle(split_results: pd.DataFrame,
                               shuffle_results: pd.DataFrame,
                               metric: str = 'rank_ic',
                               figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        actual = split_results[metric].values
        shuffled = shuffle_results[metric].values

        ax = axes[0]
        x = np.arange(len(actual))
        width = 0.35

        ax.bar(x - width / 2, actual, width, label='Actual', alpha=0.8, color='steelblue')
        ax.bar(x + width / 2, shuffled, width, label='Shuffled', alpha=0.8, color='coral')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Split')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.upper()}: Actual vs Shuffled')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(actual, bins=15, alpha=0.6,
                label=f"Actual (μ={actual.mean():.3f})",
                color='steelblue', edgecolor='black')
        ax.hist(shuffled, bins=15, alpha=0.6,
                label=f"Shuffled (μ={shuffled.mean():.3f})",
                color='coral', edgecolor='black')

        ax.axvline(actual.mean(), color='steelblue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(shuffled.mean(), color='coral', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_metric_timeline(split_results: pd.DataFrame,
                             metrics: List[str] = ['rank_ic', 'r2', 'mse'],
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        colors = ['steelblue', 'forestgreen', 'coral', 'purple', 'orange']

        for ax, metric, color in zip(axes, metrics, colors):
            values = split_results[metric].values
            splits = split_results['split'].values

            ax.plot(splits, values, marker='o', linewidth=2, markersize=6,
                    color=color, alpha=0.7, label=metric.upper())

            mean_val = values.mean()
            ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=2,
                       alpha=0.5, label=f'Mean: {mean_val:.3f}')

            if metric in ['rank_ic', 'r2']:
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

            if metric == 'rank_ic':
                ax.axhspan(0.3, values.max(), alpha=0.1, color='green', label='Strong (>0.3)')
                ax.axhspan(0, 0.3, alpha=0.05, color='yellow')
                ax.axhspan(values.min(), 0, alpha=0.1, color='red', label='Negative')

            ax.set_xlabel('Split (Time Period)')
            ax.set_ylabel(metric.replace('_', ' ').upper())
            ax.set_title(f'{metric.replace("_", " ").title()} Over Splits')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confidence_intervals(split_results: pd.DataFrame,
                                  metric: str = 'rank_ic',
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:


        values = split_results[metric].values
        splits = split_results['split'].values

        mean = values.mean()
        sem = sp_stats.sem(values)
        ci_95 = sp_stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(splits, values, s=100, alpha=0.6, color='steelblue',
                   label='Individual splits', zorder=3)

        ax.axhline(y=mean, color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean:.3f}', zorder=2)

        ax.axhspan(ci_95[0], ci_95[1], alpha=0.2, color='steelblue',
                   label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]', zorder=1)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Split')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.upper()} with 95% Confidence Interval')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class FeatureVisualizer:

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame,
                                top_n: int = 20,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:

        fig, ax = plt.subplots(figsize=figsize)

        top_features = importance_df.head(top_n)

        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], alpha=0.8, color='steelblue')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Features by Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_groups(group_stats: Dict[str, Dict],
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:

        df = pd.DataFrame(group_stats).T
        df = df.sort_values('total_importance', ascending=True)

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['total_importance'], alpha=0.8, color='forestgreen')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df.index)
        ax.set_xlabel('Total Importance')
        ax.set_title('Feature Importance by Group')
        ax.grid(True, alpha=0.3, axis='x')

        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['total_importance'], i, f"  (n={int(row['count'])})",
                    va='center', fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cumulative_importance(importance_df: pd.DataFrame,
                                   target_threshold: float = 0.95,
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:

        fig, ax = plt.subplots(figsize=figsize)

        n_features = len(importance_df)
        cumulative = importance_df['cumulative_importance'].values

        ax.plot(range(1, n_features + 1), cumulative, linewidth=2,
                color='steelblue', label='Cumulative Importance')

        ax.axhline(y=target_threshold, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'{target_threshold:.0%} Threshold')

        n_needed = np.searchsorted(cumulative, target_threshold) + 1
        ax.axvline(x=n_needed, color='red', linestyle='--', linewidth=2, alpha=0.7)

        ax.scatter([n_needed], [target_threshold], s=100, color='red', zorder=5)
        ax.text(n_needed, target_threshold,
                f'  {n_needed} features\n  ({n_needed / n_features:.1%} of total)',
                fontsize=10, va='bottom')

        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Feature Importance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class RegimeVisualizer:

    @staticmethod
    def plot_ic_vs_regime_features(regime_df: pd.DataFrame,
                                   features: List[str] = ['market_volatility', 'market_return'],
                                   figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:

        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=figsize)

        if n_features == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):
            x = regime_df[feature]
            y = regime_df['rank_ic']

            ax.scatter(x, y, s=100, alpha=0.6, color='steelblue')

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Rank IC')
            ax.set_title(f'IC vs {feature.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_regime_comparison(regime_df: pd.DataFrame,
                               ic_threshold: float = 0.3,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:

        good = regime_df[regime_df['rank_ic'] > ic_threshold]
        bad = regime_df[regime_df['rank_ic'] < ic_threshold]

        features = ['market_volatility', 'market_return', 'cumulative_return', 'max_drawdown']

        fig, axes = plt.subplots(1, len(features), figsize=figsize)

        for ax, feature in zip(axes, features):
            good_vals = good[feature]
            bad_vals = bad[feature]

            positions = [1, 2]
            ax.boxplot([good_vals, bad_vals], positions=positions, widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

            ax.set_xticks(positions)
            ax.set_xticklabels([f'Good\n(IC>{ic_threshold})', f'Bad\n(IC<{ic_threshold})'])
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(feature.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


class ReportGenerator:

    def __init__(self, output_dir: Path):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_validation_report(self,
                                   split_results: pd.DataFrame,
                                   shuffle_results: pd.DataFrame,
                                   prefix: str = 'validation') -> List[Path]:

        saved_plots = []

        fig = ValidationVisualizer.plot_actual_vs_shuffle(split_results, shuffle_results)
        path = self.output_dir / f'{prefix}_actual_vs_shuffle.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots.append(path)

        fig = ValidationVisualizer.plot_metric_timeline(split_results)
        path = self.output_dir / f'{prefix}_timeline.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots.append(path)

        fig = ValidationVisualizer.plot_confidence_intervals(split_results)
        path = self.output_dir / f'{prefix}_confidence_intervals.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots.append(path)

        print(f"Saved {len(saved_plots)} validation plots to {self.output_dir}")
        return saved_plots
