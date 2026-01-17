import numpy as np
import pandas as pd
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, return_windows=[1,5,21,63,126], volatility_windows=[5, 21, 63, 126],momentum_windows=[5,21, 63, 126, 252],lags=[1, 5, 21]
                 , rsi_windows=[14,21], sma_windows=[50,100,200], ema_windows=[50,100,200], macd_params=[(12,26,9)]):

        self.return_windows = return_windows
        self.volatility_windows = volatility_windows
        self.momentum_windows = momentum_windows
        self.lags = lags
        self.rsi_windows = rsi_windows
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.macd_params = macd_params

    def add_log_return(self, df):
        for window in self.return_windows:
            df[f'log_return_{window}'] = np.log(df['Close'] / df['Close'].shift(window))
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

    def add_rsi(self, df):
        for window in self.rsi_windows:
            df[f'rsi_{window}'] = ta.rsi(df['Close'], length=window)
        return df

    def add_sma(self, df):
        for window in self.sma_windows:
            df[f'sma_{window}'] = ta.sma(df['Close'], length=window)
        return df

    def add_ema(self, df):
        for window in self.ema_windows:
            df[f'ema_{window}'] = ta.ema(df['Close'], length=window)
        return df

    def add_macd(self, df):
        for fast, slow, signal in self.macd_params:
            macd = ta.macd(close=df['Close'], fast=fast, slow=slow, signal=signal)
            if macd is not None:
                df[f'macd_{fast}_{slow}_{signal}'] = macd[f'MACD_{fast}_{slow}_{signal}']
                df[f'macd_signal_{fast}_{slow}_{signal}'] = macd[f'MACDs_{fast}_{slow}_{signal}']
                df[f'macd_hist_{fast}_{slow}_{signal}'] = macd[f'MACDh_{fast}_{slow}_{signal}']
        return df

    def build_features(self, df):
        df = df.copy()
        df = self.add_log_return(df)
        df = self.add_rolling_volatility(df)
        df = self.add_rolling_momentum(df)
        df = self.add_lagged_returns(df)
        df = self.add_rsi(df)
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = df.dropna()
        return df
