from config.config import RUN_ID
import sqlite3
import json
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self, db_path="../data/experiments/tracking.db", run_id=RUN_ID):
        self.run_id = run_id
        self.db_path = db_path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()

        self.setup_database()

    def setup_database(self):

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY UNIQUE,
                name TEXT,
                model_type TEXT,
                dataset_path TEXT,
                config_json TEXT,
                created_at TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                split INTEGER,
                mse REAL,
                r2 REAL,
                rank_ic REAL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(run_id) ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                summary_json TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(run_id) ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                model_path TEXT,
                predictions_path TEXT,
                plots_path TEXT,
                analysis_path TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(run_id) ON DELETE CASCADE
            )
            """
        )

        self.conn.commit()

    def start_experiment(self, config) -> int:

        model_type = config.get("model", {}).get("model_type")
        dataset_path = config.get("data", {}).get("dataset_path")
        name = config.get("experiment_name")

        self.cursor.execute(
            """
            INSERT INTO experiments (run_id, name, model_type, dataset_path, config_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                name,
                model_type,
                dataset_path,
                json.dumps(config, default=str),
                datetime.utcnow().isoformat(),
            ),
        )

        self.conn.commit()
        return self.cursor.lastrowid

    def log_split_metrics(self, split, mse, r2, rank_ic):
        self.cursor.execute(
            """
            INSERT INTO metrics (experiment_id, split, mse, r2, rank_ic)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.run_id, split, mse, r2, rank_ic),
        )
        self.conn.commit()

    def log_summary(self, summary):
        self.cursor.execute(
            """
            INSERT INTO summaries (experiment_id, summary_json)
            VALUES (?, ?)
            """,
            (self.run_id, json.dumps(summary, default=str)),
        )
        self.conn.commit()

    def log_artifacts(
            self,
            model_path=None,
            predictions_path=None,
            plots_path=None,
            analysis_path=None,
    ):
        self.cursor.execute(
            """
            INSERT INTO artifacts (
                experiment_id, model_path, predictions_path, plots_path,analysis_path
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.run_id, model_path, predictions_path, plots_path, analysis_path),
        )
        self.conn.commit()

    def get_experiment_metrics(self, experiment_id):
        self.cursor.execute(
            """
            SELECT split, mse, r2, rank_ic
            FROM metrics
            WHERE experiment_id = ?
            ORDER BY split
            """,
            (experiment_id,),
        )

        return self.cursor.fetchall()

    def list_experiments(self):

        self.cursor.execute(
            """
            SELECT id, run_id, name, model_type, dataset_path, created_at
            FROM experiments
            ORDER BY created_at DESC
            """
        )

        return self.cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
