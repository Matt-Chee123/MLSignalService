from pathlib import Path
import pandas as pd
from data.fetch_data import fetch_universe
from datetime import datetime, date, timedelta

def format_date(date):
    dt_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
    return dt_object.strftime("%Y-%m-%d")

def load_backtest_data(experiment, run_id):
    prediction_loader = PredictionLoader(experiment)
    market_loader = MarketDataLoader()

    pred_data = prediction_loader.load_from_csv(run_id)
    raw_start = datetime.strptime(pred_data['Date'].min(), "%Y-%m-%d %H:%M:%S%z")
    raw_end = datetime.now()

    start = raw_start.strftime("%Y-%m-%d")
    end = raw_end.strftime("%Y-%m-%d")

    print(f"Tickers: {pred_data['Ticker'].unique().tolist()}")
    print(f"Range: {start} to {end}")

class PredictionLoader:
    def __init__(self, experiment='rf_signal_v1'):
        self.experiment_path = Path('../training/artifacts') / experiment

    def load_from_csv(self, run):
        data = pd.read_csv(self.experiment_path / run / 'predictions/predictions.csv')
        return data

class MarketDataLoader:
    def __init__(self):
        pass

    def fetch_historical_data(self, tickers, start, end):
        data = fetch_universe(tickers, start, end)
        return data

    def compute_forward_returns(self, prices, horizon=60):
        if not isinstance(prices.index, pd.MultiIndex):
            raise ValueError("Prices must have MultiIndex (Date, Ticker)")

        df = prices.copy()

        df[f"forward_return_{horizon}d"] = (
                df.groupby(level="Ticker")['Adj Close']
                .shift(-horizon) / df['Adj Close'] - 1
        )

        return df

load_backtest_data('rf_signal_v1', '20260219_210026')