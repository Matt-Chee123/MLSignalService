from evidently import Dataset, DataDefinition, Report, Regression
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift

class DriftMonitor:
    def __init__(self, config, reference, current, metadata):
        self.config = config
        self.metadata = metadata
        self.reference = reference
        self.current = current
        self.feature_names = metadata['feature_names']
        self.prediction_col = metadata.get('prediction_column', 'prediction')

        self.feature_definition = self._build_definition(self.feature_names)
        self.prediction_definition = self._build_definition([self.prediction_col])
        self.reference_ds = self._build_dataset(self.reference)
        self.current_ds = self._build_dataset(self.current)

    def _build_definition(self, numerical_columns=None, categorical_columns=None):
        return DataDefinition(numerical_columns=numerical_columns, categorical_columns=categorical_columns)

    def _build_dataset(self, data):
        return Dataset.from_pandas(data)

    def detect_feature_drift(self):
        metrics = [ValueDrift(column=feature, method='psi') for feature in self.feature_names]
        report = Report(metrics=metrics)
        snapshot = report.run(reference_data=self.reference_ds, current_data=self.current_ds)

        result = snapshot.dict()

        psi_threshold = self.config.get('feature_psi_threshold', 0.25)

        parsed = {}
        for metric in result['metrics']:
            feature = metric['config']['column']
            score = metric['value']
            parsed[feature] = {
                'method': 'psi',
                'drift_score': score,
                'drifted': score > psi_threshold,
            }
        return parsed