from pathlib import Path
import pandas as pd

class PredictionLoader:
    def __init__(self, experiment='rf_signal_v1'):
        self.experiment_path = Path('../training/artifacts') / experiment

    def load_from_csv(self, run):
        data = pd.read_csv(self.experiment_path / run / 'predictions/predictions.csv')
        print(data.head())
        return data

loader = PredictionLoader()
loader.load_from_csv('20260219_210026')