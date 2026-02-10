import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RegimeAnalysis:
    regime_df: pd.DataFrame
    correlations: pd.Series
    good_regime_characteristics: Dict
    bad_regime_characteristics: Dict
    regime_clusters: Optional[pd.DataFrame] = None


class ModelDiagnostics:
    def __init__(self, split_results, shuffle_results):
        self.split_results = split_results
        self.shuffle_results = shuffle_results

    def analyse_stability(self, metric = 'rank_ic'):
        values = split