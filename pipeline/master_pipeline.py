from data_pipeline import MarketDataPipeline
from analysis.orchestrator import AnalysisOrchestrator
from training.orchestrator import TrainingOrchestrator
from config.config import TRAINING_CONFIG, RUN_ID
from backtest.backtest_orchestrator import BacktestOrchestrator

def run_pipeline(training_config=TRAINING_CONFIG):

    data_pipeline = MarketDataPipeline()
    training_pipeline = TrainingOrchestrator(training_config)

    output_dir = data_pipeline.run()
    training_pipeline.run_pipeline(output_dir)

    analysis_pipeline = AnalysisOrchestrator(training_pipeline)

    analysis_pipeline.run_full_analysis()

    backtest_orch = BacktestOrchestrator(training_config)
    backtest_orch.run()

run_pipeline()