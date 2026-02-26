from pathlib import Path
import pandas as pd

from config.backtest_config import HORIZON, BENCHMARK
from data.fetch_data import fetch_universe
from datetime import datetime, date, timedelta

def format_date(date):
    dt_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
    return dt_object.strftime("%Y-%m-%d")

def load_backtest_data(experiment, run_id, horizon):
    prediction_loader = PredictionLoader(experiment)
    market_loader = MarketDataLoader(BENCHMARK)

    pred_data = prediction_loader.load_from_csv(run_id)

    pred_data['Date'] = pd.to_datetime(pred_data['Date'])

    pred_data = pred_data.set_index(['Date', 'Ticker']).sort_index()

    tickers = pred_data.index.get_level_values('Ticker').unique().tolist()

    raw_start = pred_data.index.get_level_values('Date').min()
    raw_end = datetime.now()

    start = raw_start.strftime("%Y-%m-%d")
    end = raw_end.strftime("%Y-%m-%d")

    market_data = market_loader.fetch_historical_data(tickers, start, end)
    forward_returns = market_loader.compute_forward_returns(market_data, HORIZON)


    return {
        'predictions': pred_data,
        'prices': market_data,
        'returns': forward_returns
    }

class PredictionLoader:
    def __init__(self, experiment='rf_signal_v1'):
        self.experiment_path = Path('../training/artifacts') / experiment

    def load_from_csv(self, run):
        data = pd.read_csv(self.experiment_path / run / 'predictions/predictions.csv')
        return data

class MarketDataLoader:
    def __init__(self, benchmark='^GSPC'):
        self.benchmark = benchmark

    def fetch_historical_data(self, tickers, start, end):
        tickers.append(self.benchmark)
        data = fetch_universe(tickers, start, end)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(level=1)
            data.index.names = ['Date', 'Ticker']
        data = data.set_index(['Date','Ticker'])

        return data

    def compute_forward_returns(self, prices, horizon=3):
        if not isinstance(prices.index, pd.MultiIndex):
            raise ValueError("Prices must have MultiIndex (Date, Ticker)")

        horizon = horizon * 20

        df = prices.copy()
        df["forward_return"] = (
                df.groupby(level="Ticker")['Close']
                .shift(-horizon) / df['Close'] - 1
        )

        return df

