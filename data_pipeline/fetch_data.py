import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    logger.info(f"Fetching data for ticker={ticker}")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        info = stock.info
        fundamentals = {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "debt_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "sector": info.get("sector"),
            "rev_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "fcf": info.get("freeCashflow")
        }

        if df.empty:
            logger.warning(f"No data returned for ticker={ticker}")
            return None

        df = df.reset_index()

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Ticker"] = ticker

        for key, value in fundamentals.items():
            df[key] = value

        return df

    except Exception as e:
        logger.exception(f"Failed to fetch data for ticker={ticker}: {e}")
        return None


def fetch_universe(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    dfs = []

    for ticker in tickers:
        df = fetch_stock_data(ticker, start, end, interval)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No data fetched for any ticker")

    data = pd.concat(dfs, ignore_index=True)
    data.sort_values(["Ticker", "Date"], inplace=True)

    logger.info(
        f"Fetched data for {data['Ticker'].nunique()} tickers "
        f"({len(data):,} rows)"
    )

    return data


def save_data(
    df: pd.DataFrame,
    output_path: str
) -> None:

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker, ticker_df in df.groupby("Ticker"):
        filepath = output_dir / f"{ticker}.csv"
        ticker_df.drop(columns="Ticker").to_csv(filepath, index=False)
        logger.info(f"Saved raw data for {ticker} → {filepath}")

