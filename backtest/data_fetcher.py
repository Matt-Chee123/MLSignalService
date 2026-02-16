from pathlib import Path
from config.backtest_config import RUN_ID
import pandas as pd
import yfinance as yf

class MarketDataFetcher:
    def __init__(self, run_id=RUN_ID, processed_path='../data/processed'):
        self.run_id = run_id
        self.processed_path = Path(processed_path) / str(run_id)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.data_file = self.processed_path / 'data.csv'

    def load_csv(self):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        data = pd.read_csv(self.data_file, parse_dates=['Date'], index_col=['Date', 'Ticker'])
        print(f"Loaded {len(data)} rows from {self.data_file}")
        return data

    def save_csv(self, df):
        df.to_csv(self.data_file)
        print(f"Saved {len(df)} rows to {self.data_file}")

    def fetch_yfinance(self, tickers, start_date, end_date):
        all_data = []

        for ticker in tickers:
            print(f"Fetching {ticker} from {start_date} to {end_date}")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df['Ticker'] = ticker
            all_data.append(df.reset_index())

        df_all = pd.concat(all_data)
        df_all = df_all[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_all.set_index(['Date', 'Ticker'], inplace=True)
        df_all.sort_index(inplace=True)
        return df_all
