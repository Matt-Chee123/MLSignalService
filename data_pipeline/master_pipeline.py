from data_pipeline.data_orchestrator import MarketDataPipeline
from analysis.orchestrator import AnalysisOrchestrator
from training.orchestrator import TrainingOrchestrator
from config.config import TRAINING_CONFIG, RUN_ID
from backtest.backtest_orchestrator import BacktestOrchestrator

import json
from datetime import datetime

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = timestamp

    return config

def run_pipeline(training_config=TRAINING_CONFIG):

    data_pipeline = MarketDataPipeline(tickers=training_config['tickers'], run_id=training_config['run_id'])
    training_pipeline = TrainingOrchestrator(training_config)

    output_dir = data_pipeline.run()
    training_pipeline.run_pipeline(output_dir)

    analysis_pipeline = AnalysisOrchestrator(training_pipeline)

    analysis_pipeline.run_full_analysis()

    backtest_orch = BacktestOrchestrator(training_config)
    backtest_orch.run()

    return training_pipeline, analysis_pipeline, backtest_orch

run_pipeline()