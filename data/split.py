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

        possible_iterations = len(unique_dates) - self.horizon - self.window_size + 1

        iterations = min(self.iterations, possible_iterations)

        splits = []

        for i in range(iterations):
            train_split = unique_dates[i: i+self.window_size]
            test_split = unique_dates[i + self.window_size : i + self.window_size + self.horizon]
            train = data.loc[data.index.get_level_values('date').isin(train_split)]
            test = data.loc[data.index.get_level_values('date').isin(test_split)]
            splits.append((train, test))

        return splits


