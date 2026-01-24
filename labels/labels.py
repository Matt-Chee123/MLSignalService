import numpy as np
import pandas as pd

class LabelGenerator:
    def __init__(self, horizon=126):
        self.horizon = horizon

    def add_return_label(self, df):
        df['return_label'] = df.groupby('ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        return df

    def add_binary_label(self, df):
        pct_change = df.groupby('ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        df['binary_label'] = (pct_change > 0).astype(float).where(pct_change.notna())
        return df

    def add_relative_rank(self, df):
        df = self.add_return_label(df)
        df['rank_label'] = df.groupby(level='date')['return_label'].rank(pct=True)
        df = df.drop(['return_label'], axis=1)
        return df
