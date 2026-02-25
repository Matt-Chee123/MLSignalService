from backtest.data_loaders import load_backtest_data
from backtest.align_data import DataAligner
from backtest.portfolio import PortfolioConstructor
from config.backtest_config import HORIZON

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

@dataclass
class BacktestResults:
    period_returns: pd.Series
    cumulative_returns: pd.Series
    metrics: Dict[str, float]

    def print_summary(self):
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        for metric, value in self.metrics.items():
            if "Ratio" in metric:
                print(f"{metric:.<40} {value:.2f}")
            else:
                print(f"{metric:.<40} {value:.2%}")
        print("=" * 60 + "\n")


class SimpleBacktest:
    def __init__(
        self, portfolio, horizon_months=3, risk_free_rate=0.02, initial_value=100000):

        self.portfolio = portfolio.copy()
        self.horizon_months = horizon_months
        self.periods_per_year = 12 // horizon_months
        self.risk_free = risk_free_rate
        self.initial_value = initial_value

    def run(self):
        period_returns = self._compute_period_returns()
        cumulative_returns = (1 + period_returns).cumprod()

        metrics = self._compute_metrics(period_returns, cumulative_returns)
        return BacktestResults(
            period_returns=period_returns,
            cumulative_returns=cumulative_returns,
            metrics=metrics,
        )

    def _compute_period_returns(self) -> pd.Series:
        """
        Compute non-overlapping period returns based on portfolio weights
        and forward returns. The horizon is in months (self.horizon_months).
        """
        df = self.portfolio.copy()

        # Ensure weights are in 0-1 range
        if df["weight"].abs().max() > 1:
            df["weight"] /= 100

        # Sort by date
        df = df.sort_index()

        all_dates = df.index.get_level_values("Date").unique().sort_values()

        step = self.horizon_months * 20

        rebalance_dates = all_dates[::step]

        period_returns_list = []

        for date in rebalance_dates:
            daily_portfolio = df.loc[date]

            weighted_ret = (daily_portfolio["weight"] * daily_portfolio["forward_return"]).sum()
            period_returns_list.append((date, weighted_ret))

        period_returns = pd.Series(
            [r for _, r in period_returns_list],
            index=[d for d, _ in period_returns_list],
            name="period_return"
        )

        return period_returns

    def _compute_metrics(self, returns, cumulative):

        mean_return = returns.mean()
        vol = returns.std()

        ann_return = (1 + mean_return) ** self.periods_per_year - 1
        ann_vol = vol * np.sqrt(self.periods_per_year)

        sharpe = (
            (ann_return - self.risk_free) / ann_vol
            if ann_vol != 0 else 0
        )

        max_dd = self._max_drawdown(cumulative)

        turnover = self._compute_turnover()
        gross_exposure = self._compute_gross_exposure()

        return {
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Mean Period Return": mean_return,
            "Average Turnover": turnover,
            "Average Gross Exposure": gross_exposure,
        }

    def _max_drawdown(self, cumulative):

        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1

        return drawdown.min()

    def _compute_turnover(self):

        df = self.portfolio.copy()

        if df["weight"].abs().max() > 1:
            df["weight"] /= 100

        df["prev_weight"] = (
            df.groupby(level="Ticker")["weight"]
            .shift(1)
        )

        turnover = (
            (df["weight"] - df["prev_weight"])
            .abs()
            .groupby(level="Date")
            .sum()
        )

        return turnover.mean()

    def _compute_gross_exposure(self):

        df = self.portfolio.copy()

        if df["weight"].abs().max() > 1:
            df["weight"] /= 100

        gross = (
            df["weight"]
            .abs()
            .groupby(level="Date")
            .sum()
        )

        return gross.mean()


data = load_backtest_data('rf_signal_v1', '20260224_202528', HORIZON)
aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
combined_data = aligner.align()


constructor = PortfolioConstructor(strategy='quantile_long_short')
portfolio = constructor.construct(combined_data)

backtester = SimpleBacktest(portfolio)

metrics = backtester.run()
print(metrics)
metrics.print_summary()