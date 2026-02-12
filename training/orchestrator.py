import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from training.loader import load_splits, load_metadata
from training.trainer import Trainer
from training.evaluator import Evaluator
from training import metrics as metric_lib
from models.model_factory import get_model_from_config
from config.config import TRAINING_CONFIG
from analysis.validators import StatisticalValidator, ValidationResults, RegimeAnalyser
from analysis.diagnostics import ModelDiagnostics

class TrainingOrchestrator:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config['training']
        self.evaluator = Evaluator(
            metrics={
                name: getattr(metric_lib, name)
                for name in config.get("metrics", [])
            }
        )

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = config.get("experiment_name", "default_experiment")

        self.base_output_dir = Path(
            self.training_config.get("output_dir", "./artifacts")
        )
        self.run_dir = self.base_output_dir / self.experiment_name / self.run_id
        self.models_dir = self.run_dir / "models"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"

        for d in [self.run_dir, self.models_dir, self.metrics_dir, self.logs_dir]:
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

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        fh = logging.FileHandler(self.logs_dir / "run.log")
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

        return logger

    def _save_config_state(self):
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def load_data(self):
        self.splits = load_splits(self.data_config['dataset_path'])
        self.metadata = load_metadata(self.data_config['dataset_path'])

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

    def run_cross_validation(self):
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
            self.split_results.append({
                "split": idx,
                **metrics
            })
            self.logger.info(f"Split {idx} | {metrics}")
        self.summary_metrics = self.evaluator.evaluate_all_splits(self.split_results)
        self.logger.info(f"Summary metrics: {self.summary_metrics}")

orch = TrainingOrchestrator(TRAINING_CONFIG)
orch.load_data()
orch.run_cross_validation()
orch.run_shuffle_test()
val = RegimeAnalyser(orch.split_results['split'], orch.split_results)
val.print_regime_report()
