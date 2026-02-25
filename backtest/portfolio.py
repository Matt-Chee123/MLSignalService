from backtest.data_loaders import load_backtest_data
from backtest.align_data import DataAligner
from config.backtest_config import HORIZON

import pandas as pd

class PortfolioConstructor:
    def __init__(self, strategy='long_only', long_pct=0.2, short_pct=0.2, sizing='equal', neutralize='market'):
        self.strategy = strategy
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.sizing = sizing
        self.neutralize = neutralize

        self.portfolio = None

    def _long_only(self, data):
        portfolios = []

        for date, group in data.groupby('Date'):
            n = len(group)

            long_n = int(n * self.long_pct)
            short_n = int(n * self.short_pct)
            sorted_group = group.sort_values('pred_label', ascending=True)
            longs = sorted_group.tail(long_n).copy()
            longs['weight'] = 1 / long_n
            daily_portfolio = pd.DataFrame(longs)
            portfolios.append(daily_portfolio)

        return pd.concat(portfolios)

    def _quantile_long_short(self, data):

        portfolios = []

        for date, group in data.groupby('Date'):
            n = len(group)

            long_n = int(n * self.long_pct)
            short_n = int(n * self.short_pct)
            sorted_group = group.sort_values('pred_label', ascending=True)
            longs = sorted_group.tail(long_n).copy()
            shorts = sorted_group.head(short_n).copy()
            longs['weight'] = 0.5 / long_n
            shorts['weight'] = -0.5 / short_n

            daily_portfolio = pd.concat([longs, shorts])
            portfolios.append(daily_portfolio)

        return pd.concat(portfolios)

    def construct(self, aligned_data):
        if aligned_data is None or aligned_data.empty:
            raise ValueError("No aligned data found")


        if self.strategy == 'quantile_long_short':
            portfolio = self._quantile_long_short(aligned_data)
        elif self.strategy == 'long_only':
            portfolio = self._long_only(aligned_data)
        else:
            raise ValueError("No correct strategy found")
        return portfolio


data = load_backtest_data('rf_signal_v1', '20260224_202528', HORIZON)
aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
combined_data = aligner.align()

constructor = PortfolioConstructor()
constructor.construct(combined_data)