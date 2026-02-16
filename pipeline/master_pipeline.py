from data_pipeline import MarketDataPipeline
from analysis.orchestrator import AnalysisOrchestrator
from training.orchestrator import TrainingOrchestrator
from config.config import TRAINING_CONFIG

def run_pipeline():
    data_pipeline = MarketDataPipeline()
    training_pipeline = TrainingOrchestrator(TRAINING_CONFIG)

    output_dir = data_pipeline.run()
    training_pipeline.run_pipeline(output_dir)

    analysis_pipeline = AnalysisOrchestrator(training_pipeline)

    analysis_pipeline.run_full_analysis()

run_pipeline()