import sqlite3
from pathlib import Path
import json
import pandas as pd

class CompareModels:
    def __init__(self, experiment_ids=[], db_path='../data/experiments/tracking.db'):
        self.experiment_ids = experiment_ids
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

