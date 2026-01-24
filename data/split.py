import pandas as pd
import numpy as np

class DataSplitter:
    def __init__(self, window_size=2, horizon=3, iterations=15):
        self.window_size = window_size # * 252
        self.horizon = horizon #* 20
        self.iterations = iterations

    def split(self, data):
        data = data.sort_index()

        if not isinstance(data.index, pd.MultiIndex):
            raise TypeError(
                f"Expected a MultiIndex (Date, Ticker), but got {type(data.index)}. "
                "Please ensure your DataFrame is indexed correctly."
            )

        unique_dates = data.index.get_level_values(0).unique().values

        if len(unique_dates)  <  (self.window_size + self.horizon):
            raise ValueError(
                f"Data length ({len(unique_dates)}) is too short for "
                f"window_size {self.window_size} and horizon {self.horizon}."
            )

        print(unique_dates)
        return 0, 0

dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10))
tickers = ['AAPL', 'MSFT', 'GOOG']

midx = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
np.random.seed(42)
test_df = pd.DataFrame(
    np.random.rand(30, 2),
    index=midx,
    columns=['Close', 'Feature1']
)

splitter = DataSplitter(3,2, 3)
train_split, test_split = splitter.split(test_df)

