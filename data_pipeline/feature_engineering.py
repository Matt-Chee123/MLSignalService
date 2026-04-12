import numpy as np
import pandas as pd
import pandas_ta as ta
import json

class FeatureEngineer:
    def __init__(self, config):

        self.return_windows = config['return_windows']
        self.volatility_windows = config['volatility_windows']
        self.momentum_windows = config['momentum_windows']
        self.lags = config['lags']
        self.rsi_windows = config['rsi_windows']
        self.sma_windows = config['sma_windows']
        self.ema_windows = config['ema_windows']
        self.macd_params = [tuple(x) for x in config["macd_params"]]

    def add_log_return(self, df):
        for window in self.return_windows:
            df[f'log_return_{window}'] = df.groupby('Ticker', group_keys=False)['Close'].transform(
                lambda x: np.log(x / x.shift(window))
            )
        return df

    def add_rolling_volatility(self, df):
        daily_returns = df.groupby('Ticker', group_keys=False)['Close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        for window in self.volatility_windows:
            df[f'rolling_volatility_{window}'] = (
                daily_returns.groupby(df['Ticker'], group_keys=False)
                .transform(lambda x: x.rolling(window).std())
            )
        return df

    def add_rolling_momentum(self, df):
        for window in self.momentum_windows:
            df[f'rolling_momentum_{window}'] = df.groupby('Ticker', group_keys=False)['Close'].transform(
                lambda x: x / x.shift(window) - 1
            )
        return df

    def add_lagged_returns(self, df):
        daily_returns = df.groupby('Ticker', group_keys=False)['Close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        for lag in self.lags:
            df[f'lagged_returns_{lag}'] = (
                daily_returns.groupby(df['Ticker'], group_keys=False)
                .transform(lambda x: x.shift(lag))
            )
        return df

    def add_rsi(self, df):
        for window in self.rsi_windows:
            df[f'rsi_{window}'] = df.groupby('Ticker', group_keys=False)['Close'].transform(
                lambda x: ta.rsi(x, length=window)
            )
        return df

    def add_sma(self, df):
        for window in self.sma_windows:
            df[f'sma_{window}'] = df.groupby('Ticker', group_keys=False)['Close'].transform(
                lambda x: ta.sma(x, length=window) / x
            )
        return df

    def add_ema(self, df):
        for window in self.ema_windows:
            df[f'ema_{window}'] = df.groupby('Ticker', group_keys=False)['Close'].transform(
                lambda x: ta.ema(x, length=window) / x
            )
        return df

    def add_macd(self, df):
        for fast, slow, signal in self.macd_params:
            df[f'macd_{fast}_{slow}_{signal}'] = np.nan
            df[f'macd_signal_{fast}_{slow}_{signal}'] = np.nan
            df[f'macd_hist_{fast}_{slow}_{signal}'] = np.nan

            for ticker in df['Ticker'].unique():
                ticker_mask = df['Ticker'] == ticker
                ticker_close = df.loc[ticker_mask, 'Close']

                macd_result = ta.macd(close=ticker_close, fast=fast, slow=slow, signal=signal)

                if macd_result is not None and not macd_result.empty:
                    macd_col = f'MACD_{fast}_{slow}_{signal}'
                    signal_col = f'MACDs_{fast}_{slow}_{signal}'
                    hist_col = f'MACDh_{fast}_{slow}_{signal}'

                    if macd_col in macd_result.columns:
                        df.loc[ticker_mask, f'macd_{fast}_{slow}_{signal}'] = macd_result[macd_col].values
                    if signal_col in macd_result.columns:
                        df.loc[ticker_mask, f'macd_signal_{fast}_{slow}_{signal}'] = macd_result[signal_col].values
                    if hist_col in macd_result.columns:
                        df.loc[ticker_mask, f'macd_hist_{fast}_{slow}_{signal}'] = macd_result[hist_col].values

        return df

    def add_sector_zscores(self, df):
        target_metrics = ['pe_ratio', 'pb_ratio', 'rev_growth', 'profit_margins', 'roe', 'debt_equity', 'fcf']
        for metric in target_metrics:
            if metric not in df.columns:
                continue

            df[metric] = pd.to_numeric(df[metric], errors="coerce")

            group = df.groupby(['Date', 'sector'])[metric]

            m = group.transform('mean')
            s = group.transform('std')

            df[f'{metric}_sector_z'] = (df[metric] - m) / (s + 1e-6)
        return df

    def add_market_context(self, df):
        df['log_market_cap'] = np.log(df['market_cap'] + 1)

        df['market_cap_percentile'] = df.groupby('Date')['market_cap'].rank(pct=True)
        return df

    def build_features(self, df):
        df = df.copy()

        if isinstance(df.index, pd.MultiIndex):
            if df.index.names != ['Date', 'Ticker']:
                raise ValueError(f"Expected MultiIndex with ['Date', 'Ticker'], got {df.index.names}")
            df = df.reset_index()
        else:
            raise ValueError("Expected MultiIndex, got regular Index")

        df = df.sort_values(['Ticker', 'Date'])

        print(f"Building features for {df['Ticker'].nunique()} tickers...")

        df = self.add_log_return(df)
        df = self.add_rolling_volatility(df)
        df = self.add_rolling_momentum(df)
        df = self.add_lagged_returns(df)
        df = self.add_rsi(df)
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_sector_zscores(df)
        df = self.add_market_context(df)
        df = df.drop(columns=['sector'])

        df = df.set_index(['Date', 'Ticker']).sort_index()

        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)

        print(f"Features built: {initial_rows:,} → {final_rows:,} rows after dropping NaN")

        return df