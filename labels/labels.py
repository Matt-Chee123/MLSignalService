import numpy as np
import pandas as pd

class LabelGenerator:
    def __init__(self, horizon=3, label_type='RelRank'):
        self.horizon = horizon * 20
        self.label_type = label_type

    def add_return_label(self, df):
        df['label'] = df.groupby('Ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        return df

    def add_binary_label(self, df):
        pct_change = df.groupby('Ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        df['label'] = (pct_change > 0).astype(float).where(pct_change.notna())
        return df

    def add_relative_rank(self, df):
        df = self.add_return_label(df)
        df['label'] = df.groupby(level='Date')['label'].rank(pct=True)
        return df

    def add_labels(self, df):
        if self.label_type == 'RelRank':
            return self.add_relative_rank(df)
        elif self.label_type == 'ReturnRank':
            return self.add_return_label(df)
        else:
            return self.add_binary_label(df)