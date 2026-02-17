import numpy as np
import pandas as pd

class LabelGenerator:
    def __init__(self, horizon=3, label_type='RelRank'):
        self.horizon = horizon * 20
        self.label_type = label_type

    def add_return_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['label'] = df.groupby('Ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        return df

    def add_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        future_return = df.groupby('Ticker')['Close'].shift(-self.horizon) / df['Close'] - 1
        df['label'] = (future_return > 0).astype(float).where(future_return.notna())
        return df

    def add_relative_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['future_return'] = df.groupby('Ticker')['Close'].shift(-self.horizon) / df['Close'] - 1

        df['label'] = df.groupby(level='Date')['future_return'].rank(pct=True)

        df.drop(columns=['future_return'], inplace=True)
        return df

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_type == 'RelRank':
            return self.add_relative_rank(df)
        elif self.label_type == 'ReturnRank':
            return self.add_return_label(df)
        elif self.label_type == 'Binary':
            return self.add_binary_label(df)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
