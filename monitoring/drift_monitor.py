from evidently import Dataset, DataDefinition, Report, Regression
from evidently.metrics import ValueDrift
from datetime import date, timedelta
import uuid

class DriftMonitor:
    def __init__(self, config, reference, current, metadata, model_name):
        self.training_tickers = config['tickers']
        self.model_name = model_name

        if self.training_tickers:
            in_dist_mask = current["ticker"].isin(self.training_tickers)
            self.current_in_dist = current[in_dist_mask]
            self.current_oos = current[~in_dist_mask]
        else:
            self.current_in_dist = current
            self.current_oos = current.iloc[0:0]

        self.config = config
        self.psi_threshold = self.config.get('feature_psi_threshold', 0.25)
        self.oos_threshold = self.config.get('oos_rate_threshold', 0.05)
        self.metadata = metadata
        self.reference = reference
        self.current = current
        self.feature_names = metadata['feature_names']
        self.prediction_col = metadata.get('prediction_column', 'prediction')

        self.feature_definition = self._build_definition(self.feature_names)
        self.prediction_definition = self._build_definition([self.prediction_col])
        self.reference_ds = self._build_dataset(self.reference)
        self.current_ds = self._build_dataset(self.current_in_dist)

    def _build_definition(self, numerical_columns=None, categorical_columns=None):
        return DataDefinition(numerical_columns=numerical_columns, categorical_columns=categorical_columns)

    def _build_dataset(self, data):
        return Dataset.from_pandas(data)

    def detect_feature_drift(self):
        metrics = [ValueDrift(column=feature, method='psi') for feature in self.feature_names]
        report = Report(metrics=metrics)
        snapshot = report.run(reference_data=self.reference_ds, current_data=self.current_ds)

        result = snapshot.dict()

        parsed = {}
        for metric in result['metrics']:
            feature = metric['config']['column']
            score = metric['value']
            parsed[feature] = {
                'method': 'psi',
                'drift_score': score,
                'drifted': score > self.psi_threshold,
            }
        return parsed

    def detect_prediction_drift(self):
        metric = [ValueDrift(column=self.prediction_col, method='psi')]
        report = Report(metrics=metric)
        snapshot = report.run(current_data=self.current_ds, reference_data=self.reference_ds)

        result = snapshot.dict()

        parsed = {}
        for metric in result['metrics']:
            feature = metric['config']['column']
            score = metric['value']
            parsed[feature] = {
                'method': 'psi',
                'drift_score': score,
                'drifted': score > self.psi_threshold,
            }
        return parsed

    def detect_coverage_drift(self):
        total = len(self.current)
        oos = len(self.current_oos)
        return {
            "total_predictions": total,
            "in_distribution": total - oos,
            "out_of_distribution": oos,
            "oos_rate": oos / total if total else 0.0,
        }

    def evaluate_drift_alerts(self, feature_drift, prediction_drift, coverage_drift):
        alerts = []

        for feature, metrics in feature_drift.items():
            if metrics['drift_score'] >= self.psi_threshold:
                alerts.append({
                    'severity': 'high',
                    'type': 'feature_drift',
                    'feature': 'feature',
                    'metric': 'psi',
                    'value': metrics['drift_score'],
                    'threshold': self.psi_threshold
                })

        for column, metrics in prediction_drift.items():
            if metrics['drift_score'] >= self.psi_threshold:
                alerts.append({
                    'severity': 'high',
                    'type': 'prediction_drift',
                    'column': column,
                    'metric': 'psi',
                    'value': metrics['drift_score'],
                    'threshold': self.psi_threshold,
                })

        if coverage_drift['oos_rate'] >= self.oos_threshold:
            alerts.append({
                'severity': 'high',
                'type': 'coverage',
                'metric': 'oos_rate',
                'value': coverage_drift['oos_rate'],
                'threshold': self.oos_threshold,
            })

        severities = {a['severity'] for a in alerts}
        if 'high' in severities:
            status = 'red'
        elif 'medium' in severities:
            status = 'yellow'
        else:
            status = 'green'

        return alerts, status#

    def run(self):
        feat_drift = self.detect_feature_drift()
        pred_drift = self.detect_prediction_drift()
        cov_drift = self.detect_coverage_drift()
        alerts, status = self.evaluate_drift_alerts(feat_drift, pred_drift, cov_drift)

        run_timestamp = date.today()
        run_id = f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

        payload = {
            "run_id": run_id,
            "run_timestamp": run_timestamp.isoformat(),
            "model_name": self.model_name,
            "status": status,
            "feature_drift": feat_drift,
            "prediction_drift": pred_drift,
            "coverage_drift": cov_drift,
            "alerts": alerts,
        }

        return payload
