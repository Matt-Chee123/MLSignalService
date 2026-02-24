from backtest.data_loaders import load_backtest_data
from backtest.align_data import DataAligner
from config.backtest_config import HORIZON

class PortfolioConstructor:
    def __init__(self, strategy='quantile_long_short', long_pct=0.2, short_pct=0.2, sizing='equal', neutralize='market'):
        self.strategy = strategy
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.sizing = sizing
        self.neutralize = neutralize

        self.portfolio = None

    def _quantile_long_short(self, data):
        tickers = data.index.get_level_values('Ticker').unique().tolist()
        print(tickers)
        for date, group in data.groupby('Date'):#
            pass
        return "hewre"

    def construct(self, aligned_data):
        print("here")
        if aligned_data is None or aligned_data.empty:
            raise ValueError("No aligned data found")


        if self.strategy == 'quantile_long_short':
            print("here")
            portfolio = self._quantile_long_short(aligned_data)
        else:
            raise ValueError("No correct strategy found")
        return portfolio


data = load_backtest_data('rf_signal_v1', '20260219_210026', HORIZON)
aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
combined_data = aligner.align()

constructor = PortfolioConstructor()
constructor.construct(combined_data)