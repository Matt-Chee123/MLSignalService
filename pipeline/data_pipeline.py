import logging
import pandas as pd
from config.config import TICKERS, START_DATE, END_DATE, RAW_DATA_PATH, INTERVAL
from data.fetch_data import fetch_universe, save_raw_data
from data.split import DataSplitter
from features.feature_engineering import FeatureEngineer
from labels.labels import LabelGenerator
from data.clean import clean_market_data


class MarketDataPipeline:

    def __init__(self, tickers=TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL,
                 raw_data_path=RAW_DATA_PATH, iterations=15, logging_level=logging.INFO):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
        self.raw_data_path = raw_data_path
        self.iterations = iterations

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
        save_raw_data(raw_data, self.raw_data_path)
        return raw_data

    def process_features_and_labels(self, raw_data: pd.DataFrame):
        self.logger.info("Cleaning market data...")
        clean_data = clean_market_data(raw_data)

        self.logger.info("Building features...")
        feature_engineer = FeatureEngineer()
        feature_df = feature_engineer.build_features(clean_data)
        self.logger.info("Adding labels...")
        label_gen = LabelGenerator()
        labeled_df = label_gen.add_labels(feature_df)
        labeled_df = labeled_df.dropna(subset=['label'])
        return labeled_df

    def split_data(self, labeled_df: pd.DataFrame):
        self.logger.info(f"Splitting data into {self.iterations} iterations...")
        splitter = DataSplitter(iterations=self.iterations)
        splits = splitter.split(labeled_df)
        output_dir = splitter.save_splits(splits)
        self.logger.info(f"Created {len(splits)} train/test splits.")
        return splits, output_dir

    def run(self):
        raw_data = self.fetch_and_save()
        labeled_df = self.process_features_and_labels(raw_data)
        splits, output_dir = self.split_data(labeled_df)
        return output_dir
