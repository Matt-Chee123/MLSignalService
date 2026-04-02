import pandas as pd
import logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:

    initial_rows = len(df)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close"])

    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    invalid_price_mask = (
        (df["Open"] <= 0) |
        (df["High"] <= 0) |
        (df["Low"] <= 0) |
        (df["Close"] <= 0)
    )

    invalid_prices = int(invalid_price_mask.sum())
    if invalid_prices > 0:
        logger.warning(
            f"Dropping {invalid_prices} rows with non-positive prices"
        )
        df = df.loc[~invalid_price_mask]

    ohlc_max = df[["Open", "High", "Low", "Close"]].max(axis=1)
    ohlc_min = df[["Open", "High", "Low", "Close"]].min(axis=1)

    invalid_ohlc_mask = (df["High"] < ohlc_max) | (df["Low"] > ohlc_min)
    invalid_ohlc = int(invalid_ohlc_mask.sum())

    if invalid_ohlc > 0:
        logger.warning(
            f"Dropping {invalid_ohlc} rows with invalid OHLC relationships"
        )
        df = df.loc[~invalid_ohlc_mask]

    df.loc[df["Volume"] < 0, "Volume"] = 0

    df = (
        df.set_index(["Date", "Ticker"])
          .sort_index()
    )

    final_rows = len(df)

    logger.info(
        f"Market data cleaned: "
        f"{initial_rows:,} → {final_rows:,} rows"
    )

    return df
