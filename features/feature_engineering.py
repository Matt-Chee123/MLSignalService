import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, return_windows=[1,5,21,63,126], volatility_windows=[5, 21, 63, 126],momentum_windows=[5,21, 63, 126, 252],lags=[1, 5, 21]):

        self.return_windows = return_windows
        self.volatility_windows = volatility_windows
        self.momentum_windows = momentum_windows
        self.lags = lags

    def add_log_return(self, df):
        for window in self.return_windows:
            df[f'log_return_{window}'] = np.log(df['Close'] / df['Close'].shift(window))
            df[f'log_return_{window}'] = df[f'log_return_{window}'].fillna(0)
        return df

    def add_rolling_volatility(self, df):
        returns = np.log(df['Close'] / df['Close'].shift(1))
        for window in self.volatility_windows:
            df[f'rolling_volatility_{window}'] = returns.rolling(window).std()
        return df

    def add_rolling_momentum(self, df):
        for window in self.momentum_windows:
            df[f'rolling_momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
        return df

    def add_lagged_returns(self, df):
        returns = np.log(df['Close'] / df['Close'].shift(1))
        for window in self.lags:
            df[f'lagged_returns_{window}'] = returns.shift(window)
        return df

    def build_features(self, df):
        df = df.copy()
        df = self.add_log_return(df)
        df = self.add_rolling_volatility(df)
        df = self.add_rolling_momentum(df)
        df = self.add_lagged_returns(df)
        return df

def make_test_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")

    data = {
        "Open":   [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "High":   [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Low":    [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "Close":  [100, 101, 102, 104, 103, 105, 106, 108, 107, 109],
        "Volume": [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500],
    }

    return pd.DataFrame(data, index=dates)
