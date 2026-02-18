import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from training.loader import load_splits, load_metadata
from training.trainer import Trainer
from training.evaluator import Evaluator
from training import metrics as metric_lib
from config.config import TRAINING_CONFIG, RUN_ID
from analysis.tracking import ExperimentTracker


class TrainingOrchestrator:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config['training']
        self.tracker = ExperimentTracker(run_id=RUN_ID)

        self.evaluator = Evaluator(
            metrics={
                name: getattr(metric_lib, name)
                for name in config.get("metrics", [])
            }
        )

        self.experiment_id = self.tracker.start_experiment(self.config)

        self.run_id = RUN_ID
        self.experiment_name = config.get("experiment_name", "default_experiment")

        self.base_output_dir = Path(
            self.training_config.get("output_dir", "./artifacts")
        )
        self.run_dir = self.base_output_dir / self.experiment_name / self.run_id
        self.models_dir = self.run_dir / "models"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.train_dir = self.run_dir / "train_data"
        self.predictions_dir = self.run_dir / "predictions"

        for d in [self.run_dir, self.models_dir, self.metrics_dir, self.logs_dir, self.predictions_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = self._init_logger()

        self.splits = None
        self.split_results = []
        self.final_model = None
        self.summary_metrics = None

        self._save_config_state()

        self.logger.info(
            f"Initialized TrainingOrchestrator | "
            f"Experiment={self.experiment_name} | Run={self.run_id}"
        )

    def _init_logger(self):
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )

            fh = logging.FileHandler(self.logs_dir / "run.log")
            fh.setFormatter(formatter)

            sh = logging.StreamHandler()
            sh.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(sh)
            logger.propagate = False

        return logger

    def _save_config_state(self):
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def load_data(self, data_dir):
        self.splits = load_splits(data_dir)
        self.metadata = load_metadata(data_dir)


    def run_shuffle_test(self):
        self.shuffle_results = []
        for  idx, (train, test) in enumerate(self.splits):
            model = Trainer(self.model_config, self.models_dir)
            shuffled_train = train.copy()

            shuffled_train["label"] = (
                shuffled_train
                .groupby("Date")["label"]
                .transform(np.random.permutation)
            )

            X_train = shuffled_train.drop(columns="label")
            y_train = shuffled_train["label"]
            X_test, y_test = test.drop(columns="label"), test["label"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = self.evaluator.evaluate_split(
                y_true=y_test,
                y_pred=y_pred,
                signal=y_pred,
                future_returns=y_test
            )

            self.shuffle_results.append({
                "split": idx,
                **metrics
            })
            self.logger.info(f"Split {idx} | {metrics}")
        self.shuffle_metrics = self.evaluator.evaluate_all_splits(self.shuffle_results)
        self.logger.info(f"Summary metrics: {self.shuffle_metrics}")
        pd.DataFrame(self.shuffle_results).to_csv(self.metrics_dir / "shuffle_metrics.csv", index=False)


    def run_cross_validation(self):
        all_predictions = []
        self.split_results = []
        for  idx, (train, test) in enumerate(self.splits):
            model = Trainer(self.model_config, self.models_dir)
            X_train, y_train = train.drop(columns="label"), train["label"]
            X_test, y_test = test.drop(columns="label"), test["label"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = self.evaluator.evaluate_split(
                y_true=y_test,
                y_pred=y_pred,
                signal=y_pred,
                future_returns=y_test
            )

            self.tracker.log_split_metrics(
                idx,
                metrics.get("mse"),
                metrics.get("r2"),
                metrics.get("rank_ic"),
            )

            self.split_results.append({
                "split": idx,
                **metrics
            })

            df_preds = X_test.copy()
            df_preds['pred_label'] = y_pred
            df_preds['actual_label'] = y_test
            df_preds['split'] = idx
            all_predictions.append(df_preds)

            self.logger.info(f"Split {idx} | {metrics}")
        self.split_metrics = self.evaluator.evaluate_all_splits(self.split_results)
        self.logger.info(f"Summary metrics: {self.split_metrics}")
        pd.DataFrame(self.split_results).to_csv(self.metrics_dir / "split_metrics.csv", index=False)

        flat_predictions = pd.concat(all_predictions).reset_index()
        flat_predictions.rename(columns={'index': 'original_index'}, inplace=True)

        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        flat_predictions.to_csv(predictions_dir / "predictions.csv", index=False)

    def train_full_model(self):
        if self.splits is None or len(self.splits) == 0:
            return

        full_train = pd.concat([train for train, _ in self.splits], ignore_index=True)

        X_full = full_train.drop(columns=['label'])
        y_full = full_train['label']

        model = Trainer(self.model_config, self.models_dir)
        self.logger.info("Training on full dataset")
        model.fit(X_full, y_full)
        model.save_model()

    def run_pipeline(self,data_dir):
        self.load_data(data_dir)
        self.run_cross_validation()
        self.run_shuffle_test()
        validation = self.evaluator.pass_validation(self.split_metrics, self.shuffle_metrics)
        summary_payload = {
            "cv_summary": self.split_metrics,
            "shuffle_summary": self.shuffle_metrics,
            "passed_validation": json.loads(json.dumps(validation, default=float)),
        }
        self.tracker.log_summary(summary_payload)
        if validation:
            self.train_full_model()

        model_path = str(self.models_dir)
        predictions_path = str(self.predictions_dir)
        metrics_path = str(self.metrics_dir)

        self.tracker.log_artifacts(
            model_path=str(self.models_dir),
            predictions_path=str(self.predictions_dir),
            plots_path=str(self.metrics_dir),
        )
        self.tracker.close()

