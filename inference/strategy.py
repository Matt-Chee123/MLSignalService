import pandas as pd

class StrategyHandler:
    def __init__(self, strategy, top_n):
        self.strategy = strategy
        self.top_n = top_n

    def format_data(self, tickers, predictions):
        if isinstance(tickers, pd.Index):
            tickers = tickers.tolist()
        if isinstance(predictions, (pd.Series, pd.DataFrame)):
            predictions = predictions.values.flatten()

        predictions = [float(p) for p in predictions]

        sorted_pairs = sorted(
            zip(tickers, predictions),
            key=lambda x: x[1],
            reverse=True
        )
        data = [
            {"rank": rank, "ticker": ticker, "score": score}
            for rank, (ticker, score) in enumerate(sorted_pairs, start=1)
        ]
        return data

    def _long_only(self, data):
        longs = data[:self.top_n]
        print(longs)
        return {"long": longs, "short": []}

    def _quantile_long_short(self, data):
        n = min(self.top_n, len(data) // 2)
        if n == 0:
            return {"long": [], "short": []}
        longs = data[:n]
        shorts = data[-n:]
        return {"long": longs, "short": shorts}

    def construct(self, tickers, predictions):
        data = self.format_data(tickers, predictions)
        if self.strategy == 'long_only':
            print(data)
            return self._long_only(data)
        elif self.strategy == 'quantile_long_short':
            return self._quantile_long_short(data)

        return None