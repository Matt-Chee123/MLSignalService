from data_pipeline.fetch_data import fetch_universe
from data_pipeline.clean import clean_market_data
from data_pipeline.feature_engineering import FeatureEngineer
import pandas as pd

class DataPreprocess:
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.feature_engineer = FeatureEngineer(feature_config)

    def prepare(self, tickers):
        live_start = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        live_data = fetch_universe(
            tickers=tickers,
            start=live_start,
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            interval='1d'
        )
        cleaned_data = clean_market_data(live_data)
        features = self.feature_engineer.build_features(cleaned_data)

        latest_date = features.index.get_level_values('Date').max()

        latest_data = features.loc[latest_date]
        return latest_data

