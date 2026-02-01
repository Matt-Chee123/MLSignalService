from dataclasses import dataclass
from typing import List
import os

@dataclass(frozen=True)
class PipelineConfig:
    tickers: List[str]
    start_date: str
    end_date: str

    feature_set: str
    feature_columns: List[str]

    target: str
    target_column: str
    horizon: int

    window_size: int
    iterations: int

    output_dir: str

    @staticmethod
    def from_env() -> "PipelineConfig":
        return PipelineConfig(
            tickers=os.getenv("TICKERS", "AAPL,MSFT,GOOG,META").split(","),
            start_date=os.getenv("START_DATE", "2021-01-01"),
            end_date=os.getenv("END_DATE", "2026-02-01"),
            feature_set=os.getenv("FEATURE_SET", "v1"),
            feature_columns=os.getenv(
                "FEATURE_COLUMNS",
                "ret_1d,vol_20d,rsi_14,macd"
            ).split(","),
            target=os.getenv("TARGET", "return"),
            target_column=os.getenv("TARGET_COLUMN", "target"),
            horizon=int(os.getenv("HORIZON", "5")),
            window_size=int(os.getenv("WINDOW_SIZE", "252")),
            iterations=int(os.getenv("ITERATIONS", "20")),
            output_dir=os.getenv("OUTPUT_DIR", "data/datasets"),
        )
