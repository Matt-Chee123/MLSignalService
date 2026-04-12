import logging
import pandas as pd
from data_pipeline.fetch_data import fetch_universe, save_data
from data_pipeline.split import DataSplitter
from data_pipeline.feature_engineering import FeatureEngineer
from data_pipeline.labels import LabelGenerator
from data_pipeline.clean import clean_market_data
from config.loader import load_config
from pathlib import Path

class MarketDataPipeline:

    def __init__(self, config, logging_level=logging.INFO):
        self.features = config['features']
        self.tickers = config['tickers']
        self.start = config['data']['start_date']
        self.end = config['data']['end_date']
        self.interval = config['data']['interval']
        self.raw_data_path = Path(config['data']['raw_data_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        self.split_data_path = Path(config['data']['split_data_path'])
        self.iterations = config['data']['splits']
        self.horizon = config['horizon']

        for d in [self.raw_data_path, self.processed_data_path, self.split_data_path]:
            d.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_and_save(self):
        self.logger.info("Fetching raw market data...")
        raw_data = fetch_universe(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval
        )

        self.logger.info(f"Saving raw data to {self.raw_data_path}...")
        save_data(raw_data, self.raw_data_path)
        return raw_data

    def process_features_and_labels(self, raw_data: pd.DataFrame):
        self.logger.info("Cleaning market data...")
        clean_data = clean_market_data(raw_data)
        clean_data.to_csv(self.processed_data_path / "data.csv")
        self.logger.info("Building features...")
        feature_engineer = FeatureEngineer(self.features)
        feature_df = feature_engineer.build_features(clean_data)
        self.logger.info("Adding labels...")
        label_gen = LabelGenerator(self.horizon)
        labeled_df = label_gen.add_labels(feature_df)
        labeled_df = labeled_df.dropna(subset=['label'])
        return labeled_df

    def split_data(self, labeled_df: pd.DataFrame):
        self.logger.info(f"Splitting data into {self.iterations} iterations...")
        splitter = DataSplitter(iterations=self.iterations, horizon=self.horizon)
        splits = splitter.split(labeled_df)
        output_dir = splitter.save_splits(splits, self.split_data_path)
        self.logger.info(f"Created {len(splits)} train/test splits.")
        return splits, output_dir

    def fetch_live_snapshot(self):

        self.logger.info("Fetching live market snapshot...")

        live_data = fetch_universe(
            tickers=self.tickers,
            start=self.start,
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            interval=self.interval
        )

        live_clean = clean_market_data(live_data)

        feature_engineer = FeatureEngineer(self.features)
        live_features = feature_engineer.build_features(live_clean)

        live_dir = self.split_data_path / "live"
        live_dir.mkdir(exist_ok=True)

        live_features.to_csv(live_dir / "live_features.csv")

        self.logger.info("Live snapshot saved.")

        return live_features

    def run(self):
        raw_data = self.fetch_and_save()
        labeled_df = self.process_features_and_labels(raw_data)
        splits, output_dir = self.split_data(labeled_df)
        live_features = self.fetch_live_snapshot()
        return output_dir

if __name__ == "__main__":
    config = load_config()
    data_pipeline = MarketDataPipeline(config)
    output = data_pipeline.run()