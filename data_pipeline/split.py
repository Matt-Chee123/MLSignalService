import pandas as pd
import numpy as np

class DataSplitter:
    def __init__(self, window_size=2, horizon=3, iterations=15, output_path='.datasets'):
        self.window_size = window_size * 252
        self.horizon = horizon * 20
        self.iterations = iterations

    def split(self, data: pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex):
            raise TypeError("Expected MultiIndex (Date, Ticker)")

        if data.index.names != ["Date", "Ticker"]:
            raise ValueError(
                f"Expected index names ['Date', 'Ticker'], got {data.index.names}"
            )

        data = data.sort_index()
        dates = data.index.get_level_values("Date").unique().sort_values()
        if len(dates) < self.window_size + self.horizon:
            raise ValueError(
                f"Not enough data: {len(dates)} dates, need {self.window_size + self.horizon}"
            )

        max_possible_splits = len(dates) - self.window_size - self.horizon
        if max_possible_splits < 0:
            raise ValueError("Not enough data for one split")
        if self.iterations == 1:
            start_indices = [max_possible_splits]
        else:
            start_indices = np.linspace(
                0,
                max_possible_splits,
                num=self.iterations,
                dtype=int
            )
        splits = []
        for start_idx in start_indices:
            train_start = dates[start_idx]
            train_end = dates[start_idx + self.window_size - 1]

            test_start = dates[start_idx + self.window_size]
            test_end = dates[start_idx + self.window_size + self.horizon - 1]
            train = data.loc[(slice(train_start, train_end), slice(None))]
            test = data.loc[(slice(test_start, test_end), slice(None))]

            splits.append((train, test))

        return splits

    def save_splits(self, splits, output_dir="../data/datasets/"):

        metadata = []

        for idx, (train, test) in enumerate(splits, start=1):
            train_file = output_dir / f"train_split_{idx}.parquet"
            test_file = output_dir / f"test_split_{idx}.parquet"
            train.to_parquet(train_file)
            test.to_parquet(test_file)

            metadata.append({
                "split": idx,
                "train_start": train.index.get_level_values("Date").min(),
                "train_end": train.index.get_level_values("Date").max(),
                "test_start": test.index.get_level_values("Date").min(),
                "test_end": test.index.get_level_values("Date").max(),
                "train_rows": len(train),
                "test_rows": len(test)
            })

        metadata_file = output_dir / "splits_metadata.parquet"
        pd.DataFrame(metadata).to_parquet(metadata_file, index=False)

        print(f"Saved {len(splits)} splits to {output_dir}")

        return output_dir

