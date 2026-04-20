import logging
import pickle, json
from pathlib import Path
import pandas as pd
import numpy as np
from config.loader import load_config
from training.loader import load_splits, load_metadata, load_live_data
from training.trainer import Trainer
from training.evaluator import Evaluator
from training import metrics as metric_lib
from training.tracking import ExperimentTracker
from training.artifact_manifest import ArtifactManifest
import boto3

class TrainingOrchestrator:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.tracker = ExperimentTracker(config)

        self.s3 = boto3.client('s3')
        self.upload = config['s3']['upload']

        self.evaluator = Evaluator(
            metrics={
                name: getattr(metric_lib, name)
                for name in config.get("metrics", [])
            }
        )

        self.run_id = config['run_id']
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
        self.analysis_dir = self.run_dir / "analysis"
        self.data_dir = self.run_dir / "data"

        for d in [self.run_dir, self.models_dir, self.metrics_dir, self.logs_dir, self.predictions_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = self._init_logger()
        self.manifest = ArtifactManifest(self.run_dir, self.run_id)

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
        self.live_data = load_live_data(data_dir)


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
        self.tracker.log_metrics(
            {f"shuffle_{k}": v for k, v in self.shuffle_metrics.items()}
        )
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

            if idx == len(self.splits) - 1:
                self.feature_names = X_train.columns.tolist()
                self.last_split = {
                        'model': model.model.model,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }

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

            df_preds = X_test.copy()
            df_preds['pred_label'] = y_pred
            df_preds['actual_label'] = y_test
            df_preds['split'] = idx
            all_predictions.append(df_preds)

            self.logger.info(f"Split {idx} | {metrics}")
        self.split_metrics = self.evaluator.evaluate_all_splits(self.split_results)
        self.tracker.log_metrics(self.split_metrics)
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
        preds = model.predict(X_full)
        reference = X_full.copy()
        reference["label"] = y_full.values
        reference["prediction"] = preds
        reference.to_parquet(self.data_dir / "reference.parquet", index=False)
        self.tracker.log_model(model.model.model, X_full, preds)

    def generate_live_signal(self):
        self.logger.info("Generating live signal...")

        model = Trainer(self.model_config, self.models_dir)
        model.load_model()

        X_live = self.live_data[self.feature_names]

        preds = model.predict(X_live)

        signal_df = self.live_data.copy()
        signal_df["signal"] = preds


        signal_df.to_csv(self.predictions_dir / "live_signal.csv", index=False)

        self.logger.info("Live signal saved.")

    def push_to_s3(self, upload=True):
        if not upload:
            return

        manifest = self.manifest.load_manifest()

        bucket = self.config["s3"]["bucket"]
        prefix = f"{self.experiment_name}/{self.run_id}/"

        for name, artifact in manifest["artifacts"].items():
            if not artifact.get("upload", False):
                continue

            rel_path = artifact["path"]
            local_path = self.run_dir / rel_path

            if local_path.is_dir():
                for file in local_path.rglob("*"):
                    if file.is_file():
                        key = prefix + str(file.relative_to(self.run_dir))
                        self.s3.upload_file(str(file), bucket, key)
            else:
                key = prefix + rel_path
                self.s3.upload_file(str(local_path), bucket, key)


    def run_pipeline(self,data_dir, live_data=None):

        self.tracker.start()
        self.tracker.log_params(self.model_config['hyperparams'])
        self.load_data(data_dir)

        self.run_cross_validation()
        self.run_shuffle_test()
        validation = self.evaluator.pass_validation(self.split_metrics, self.shuffle_metrics)
        summary_payload = {
            "cv_summary": self.split_metrics,
            "shuffle_summary": self.shuffle_metrics,
            "passed_validation": json.loads(json.dumps(validation, default=float)),
        }
        if validation:
            self.train_full_model()
            if self.live_data is not None:
                self.generate_live_signal()

        model_path = str(self.models_dir)
        predictions_path = str(self.predictions_dir)
        metrics_path = str(self.metrics_dir)

        pd.DataFrame(self.split_results).to_parquet(self.metrics_dir / "split_results.parquet")

        if hasattr(self, "shuffle_results"):
            pd.DataFrame(self.shuffle_results).to_parquet(self.metrics_dir / "shuffle_results.parquet")

        last = self.last_split
        train_df = last["X_train"].copy()
        train_df["label"] = last["y_train"]
        train_df["_split"] = "train"
        test_df = last["X_test"].copy()
        test_df["label"] = last["y_test"]
        test_df["_split"] = "test"
        pd.concat([train_df, test_df]).to_parquet(self.data_dir / "last_split.parquet")

        with open(self.models_dir / "last_split_model.pkl", "wb") as f:
            pickle.dump(last["model"], f)

        metadata = {
            'feature_names': self.feature_names,
            'feature_metadata': self.config['features'],
            'passed_validation': str(validation),
            'horizon': self.config['horizon']
        }

        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        for i, (train, test) in enumerate(self.splits):
            train.to_parquet(self.data_dir / f"split_{i}_train.parquet")
            test.to_parquet(self.data_dir / f"split_{i}_test.parquet")

        with open(self.data_dir / "splits_count.json", "w") as f:
            json.dump({"n_splits": len(self.splits)}, f)

        self.manifest.build_manifest()
        self.manifest.save_manifest()
        # self.push_to_s3(self.upload)

        self.tracker.end()

if __name__ == "__main__":
    config = load_config()
    training_orchestrator = TrainingOrchestrator(config)
    training_orchestrator.run_pipeline(config['data']['split_data_path'])
