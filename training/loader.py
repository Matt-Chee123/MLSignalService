import pandas as pd
from pathlib import Path

def load_splits(data_path):
    data_dir = Path(data_path)
    splits = []

    for idx in range(1, 16):
        train_file = data_dir / f"train_split_{idx}.parquet"
        test_file = data_dir / f"test_split_{idx}.parquet"
        if train_file.exists() and test_file.exists():
            train = pd.read_parquet(train_file)
            test = pd.read_parquet(test_file)
            splits.append((train,test))
    return splits

def load_metadata(data_path):
    data_dir = Path(data_path)
    mdata_file = data_dir / "splits_metadata.parquet"
    return pd.read_parquet(mdata_file)

load_splits('../data/datasets/run_20260202_190702')