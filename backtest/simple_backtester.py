import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

import json
from pathlib import Path

@dataclass
class BacktestResults:
    period_returns: pd.Series
    cumulative_returns: pd.Series
    metrics: Dict[str, float]
    nav: pd.DataFrame

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

    def save(self, base_path: str):

        base_path = Path(base_path)
        backtest_path = base_path / "backtest"
        backtest_path.mkdir(parents=True, exist_ok=True)

        with open(backtest_path / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

        returns_df = self.nav.copy()
        returns_df.to_csv(backtest_path / "returns_timeseries.csv")

        plt.figure(figsize=(10, 6))
        self.nav[["Portfolio", "Benchmark"]].plot()
        plt.title("Cumulative Returns")
        plt.ylabel("Portfolio Value")
        plt.xlabel("Date")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(backtest_path / "cumulative_returns.png")
        plt.close()

class SimpleBacktest:
    def __init__(
        self, portfolio, close_prices, horizon_months=3, risk_free_rate=0.02, initial_value=100000, benchmark='^GSPC'):

        self.portfolio = portfolio.copy().sort_index(level=["Date", "Ticker"])
        self.horizon_months = horizon_months
        self.periods_per_year = 12 // horizon_months
        self.risk_free = risk_free_rate
        self.initial_value = initial_value
        self.close_prices = close_prices
        self.benchmark = benchmark

    def run(self):
        period_returns = self._compute_period_returns()
        cumulative_returns = (1 + period_returns).cumprod()

        nav = self._compute_portfolio_values()

        nav_returns = nav.pct_change(fill_method=None).dropna()

        metrics = self._compute_metrics(
            period_returns,
            cumulative_returns,
            nav_returns
        )

        return BacktestResults(
            period_returns=period_returns,
            cumulative_returns=cumulative_returns,
            nav=nav,
            metrics=metrics,
        )

    def _compute_benchmark_values(self, final_date, index):

        benchmark_prices = (self.close_prices.xs(self.benchmark, level="Ticker"))

        benchmark_prices = benchmark_prices.loc[index]

        benchmark_values = (benchmark_prices / benchmark_prices.iloc[0]) * self.initial_value

        return benchmark_values

    def _compute_portfolio_values(self):

        df = self.portfolio.copy().sort_index()

        if df["weight"].abs().max() > 1:
            df["weight"] /= 100

        all_dates = df.index.get_level_values("Date").unique().sort_values()

        step = self.horizon_months * 20
        rebalance_dates = all_dates[::step]

        nav_series = []
        current_value = self.initial_value

        for i in range(len(rebalance_dates)):

            start_date = rebalance_dates[i]

            if i < len(rebalance_dates) - 1:
                end_date = rebalance_dates[i + 1]
            else:
                end_date = self.close_prices.index.get_level_values("Date").max()

            weights = df.loc[start_date, "weight"]

            mask = ((self.close_prices.index.get_level_values("Date") >= start_date) & (self.close_prices.index.get_level_values("Date") < end_date))

            price_slice = self.close_prices.loc[mask]

            prices_wide = price_slice.unstack("Ticker")

            prices_wide = prices_wide[weights.index]

            prices_norm = prices_wide / prices_wide.iloc[0]

            segment_index = (prices_norm * weights).sum(axis=1)

            segment_values = segment_index * current_value

            current_value = segment_values.iloc[-1]

            nav_series.append(segment_values)

        portfolio_values = pd.concat(nav_series)

        portfolio_values = portfolio_values[~portfolio_values.index.duplicated()]

        benchmark_prices = self.close_prices.xs(self.benchmark, level="Ticker")
        benchmark_prices = benchmark_prices.loc[portfolio_values.index]

        benchmark_values = (benchmark_prices / benchmark_prices.iloc[0]) * self.initial_value

        comparison_df = pd.DataFrame({"Portfolio": portfolio_values, "Benchmark": benchmark_values})

        comparison_df["Portfolio_Return"] = comparison_df["Portfolio"].pct_change(fill_method=None)

        return comparison_df



    def _compute_period_returns(self) -> pd.Series:

        df = self.portfolio.copy()

        if df["weight"].abs().max() > 1:
            df["weight"] /= 100

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

    def _compute_metrics(self, returns, cumulative, nav_returns):

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

        portfolio_daily = nav_returns["Portfolio"]
        benchmark_daily = nav_returns["Benchmark"]

        excess_daily = portfolio_daily - benchmark_daily

        tracking_error = excess_daily.std() * np.sqrt(252)

        information_ratio = (
            excess_daily.mean() * 252 / tracking_error
            if tracking_error != 0 else 0
        )

        cov = np.cov(portfolio_daily, benchmark_daily)[0][1]
        beta = cov / benchmark_daily.var()

        ann_portfolio = portfolio_daily.mean() * 252
        ann_benchmark = benchmark_daily.mean() * 252

        alpha = ann_portfolio - (self.risk_free + beta * (ann_benchmark - self.risk_free))

        return {
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "Beta": beta,
            "Alpha": alpha,
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

