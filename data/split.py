import pandas as pd
import math

class DataSplitter:
    def __init__(self, window_size=2, horizon=3, iterations=15):
        self.window_size = window_size * 252
        self.horizon = horizon * 20
        self.iterations = iterations

    def split(self, data: pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex):
            raise TypeError("Expected MultiIndex (Date, Ticker)")

        if data.index.names != ["Date", "Ticker"]:
            raise ValueError(
                f"Expected index names ['Date', 'Ticker'], got {data.index.names}"
            )

        data = data.sort_index()
        dates = data.index.get_level_values("Date").unique().sort_values()

        if len(dates) < self.window_size + self.horizon:
            raise ValueError(
                f"Not enough data: {len(dates)} dates, need {self.window_size + self.horizon}"
            )

        max_possible_splits = len(dates) - self.window_size - self.horizon
        if self.iterations > 1:
            step_size = math.floor(max_possible_splits / (self.iterations - 1))
        else:
            step_size = max_possible_splits

        splits = []

        for i in range(self.iterations):
            start_idx = i * step_size
            if start_idx + self.window_size + self.horizon > len(dates):
                break

            train_start = dates[start_idx]
            train_end = dates[start_idx + self.window_size - 1]

            test_start = dates[start_idx + self.window_size]
            test_end = dates[start_idx + self.window_size + self.horizon - 1]

            train = data.loc[(slice(train_start, train_end), slice(None))]
            test = data.loc[(slice(test_start, test_end), slice(None))]

            splits.append((train, test))

        return splits
