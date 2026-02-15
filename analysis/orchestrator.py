from pathlib import Path
from training.orchestrator import TrainingOrchestrator
from config.config import TRAINING_CONFIG

from analysis.validators import StatisticalValidator
from analysis.diagnostics import ModelDiagnostics, RegimeAnalyser
from analysis.visualisations import ReportGenerator


class AnalysisOrchestrator:


    def __init__(self, training_orchestrator: TrainingOrchestrator):

        self.trainer = training_orchestrator
        self.analysis_dir = self.trainer.run_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.report_gen = ReportGenerator(self.trainer.metrics_dir)

    def run_statistical_validation(self):
        print("\n" + "="*70)
        print("RUNNING STATISTICAL VALIDATION")
        print("="*70)

        validator = StatisticalValidator(
            split_results=self.trainer.split_results,
            shuffle_results=self.trainer.shuffle_results if hasattr(self.trainer, 'shuffle_results') else None
        )
        validator.print_validation_report(metric='rank_ic')

        export_path = self.analysis_dir / "validation_results.json"
        validator.export_results(export_path)

        return validator

    def run_diagnostics(self):
        print("\n" + "="*70)
        print("RUNNING MODEL DIAGNOSTICS")
        print("="*70)

        diagnostics = ModelDiagnostics(
            split_results=self.trainer.split_results,
            shuffle_results=self.trainer.shuffle_results if hasattr(self.trainer, 'shuffle_results') else None
        )

        diagnostics.print_diagnostic_report()

        summary = diagnostics.generate_summary_report()

        import json
        export_path = self.analysis_dir / "diagnostics_summary.json"

        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        summary = convert_types(summary)

        with open(export_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return diagnostics

    def run_regime_analysis(self):
        print("\n" + "="*70)
        print("RUNNING REGIME ANALYSIS")
        print("="*70)

        regime_analyzer = RegimeAnalyser(self.trainer.splits,self.trainer.split_results)

        regime_analyzer.print_regime_report(ic_threshold=0.3)

        regime_df = regime_analyzer.extract_regime_features()
        export_path = self.analysis_dir / "regime_analysis.parquet"
        regime_df.to_parquet(export_path)

        return regime_analyzer

    def generate_visualizations(self):
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        import pandas as pd

        split_df = pd.DataFrame(self.trainer.split_results)
        shuffle_df = pd.DataFrame(self.trainer.shuffle_results) if hasattr(self.trainer, 'shuffle_results') else None

        if shuffle_df is not None:
            saved_plots = self.report_gen.generate_validation_report(
                split_results=split_df,
                shuffle_results=shuffle_df,
                prefix='validation'
            )

            print(f"Generated {len(saved_plots)} plots:")
            for plot_path in saved_plots:
                print(f"  - {plot_path.name}")

    def run_full_analysis(self):

        print("\n" + "🚀 " + "="*66)
        print("STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("="*68)

        results = {}

        try:
            results['validator'] = self.run_statistical_validation()
        except Exception as e:
            print(f"⚠️  Error in statistical validation: {e}")

        try:
            results['diagnostics'] = self.run_diagnostics()
        except Exception as e:
            print(f"⚠️  Error in diagnostics: {e}")

        try:
            results['regime_analyzer'] = self.run_regime_analysis()
        except Exception as e:
            print(f"⚠️  Error in regime analysis: {e}")

        try:
            self.generate_visualizations()
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")

        print("\n" + "✅ " + "="*66)
        print("ANALYSIS COMPLETE")
        print("="*68)
        print(f"\nAll results saved to: {self.trainer.run_dir}")
        print(f"  - Metrics: {self.trainer.metrics_dir}")
        print(f"  - Analysis: {self.analysis_dir}")
        print(f"  - Logs: {self.trainer.logs_dir}")

        return results


if __name__ == "__main__":
    print("Initializing training orchestrator...")
    trainer = TrainingOrchestrator(TRAINING_CONFIG)
    trainer.load_data()

    print("\nRunning cross-validation...")
    trainer.run_cross_validation()

    print("\nRunning shuffle test...")
    trainer.run_shuffle_test()

    print("\n" + "=" * 70)
    print("STARTING ANALYSIS PHASE")
    print("=" * 70)

    analyzer = AnalysisOrchestrator(trainer)
    results = analyzer.run_full_analysis()

    trainer.train_full_model()

    print("\n✅ Done! Check the output directory for all results and plots.")
