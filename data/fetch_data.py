import yfinance as yf
import pandas as pd
from pathlib import Path
import sys
sys.path.append('..')
from config.config import TICKERS, START_DATE, END_DATE, RAW_DATA_PATH, INTERVAL

def fetch_stock_data(ticker, start, end, interval='1d'):
    print("Fetching data for: ", ticker)
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        df.reset_index(inplace=True)

        if df.empty:
            print(f"WARNING: No data retrieved for {ticker}")
            return None
        columns_to_keep = ['Date','Open','High','Low','Close','Volume']
        df = df[columns_to_keep]

    except Exception as e:
        print(f"ERROR fetching {ticker}: {e}")
        return None

    return df

def save_to_csv(df, ticker, path):

    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = f"{path}{ticker}.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")


def main():
    successful = 0
    failed = []

    for ticker in TICKERS:
        ticker_data = fetch_stock_data(ticker, START_DATE, END_DATE)

        if ticker_data is not None:
            save_to_csv(ticker_data, ticker, RAW_DATA_PATH)
            successful += 1
        else:
            failed.append(ticker)
    print(f"\n{'='*50}")
    print(f"Data fetch complete!")
    print(f"Successful: {successful}/{len(TICKERS)}")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*50}")




if __name__ == "__main__":
    main()