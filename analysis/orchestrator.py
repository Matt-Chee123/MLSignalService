import pickle, json
import pandas as pd
from analysis.feature_analysis import FeatureAnalyser
from analysis.validators import StatisticalValidator
from analysis.diagnostics import ModelDiagnostics, RegimeAnalyser
from analysis.visualisations import ReportGenerator
from pathlib import Path
from config.loader import load_config


class AnalysisOrchestrator:
    def __init__(self, run_dir: str):

        self.run_dir = Path(run_dir)
        self.analysis_dir = self.run_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        metrics_dir = self.run_dir / "metrics"

        split_df = pd.read_parquet(metrics_dir / "split_results.parquet")
        self.split_results = split_df.to_dict(orient="records")

        shuffle_path = metrics_dir / "shuffle_results.parquet"
        self.shuffle_results = (
            pd.read_parquet(shuffle_path).to_dict(orient="records")
            if shuffle_path.exists() else None
        )

        with open(self.run_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

        data = pd.read_parquet(self.run_dir / "last_split.parquet")
        train = data[data["_split"] == "train"].drop(columns=["_split"])
        test  = data[data["_split"] == "test"].drop(columns=["_split"])

        with open(self.run_dir / "models" / "last_split_model.pkl", "rb") as f:
            model = pickle.load(f)

        self.last_split = {
            "model":   model,
            "X_train": train.drop(columns=["label"]),
            "y_train": train["label"],
            "X_test":  test.drop(columns=["label"]),
            "y_test":  test["label"],
        }

        self.report_gen = ReportGenerator(metrics_dir)

        with open(self.run_dir / "splits_count.json") as f:
            n = json.load(f)["n_splits"]

        self.splits = [
            (
                pd.read_parquet(self.run_dir / f"split_{i}_train.parquet"),
                pd.read_parquet(self.run_dir / f"split_{i}_test.parquet"),
            )
            for i in range(n)
        ]

    def run_statistical_validation(self):
        print("\n" + "="*70)
        print("RUNNING STATISTICAL VALIDATION")
        print("="*70)

        validator = StatisticalValidator(
            split_results=self.split_results,
            shuffle_results=self.shuffle_results
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
            split_results=self.split_results,
            shuffle_results=self.shuffle_results
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

        regime_analyzer = RegimeAnalyser(self.splits,self.split_results)

        regime_analyzer.print_regime_report(ic_threshold=0.3)

        regime_df = regime_analyzer.extract_regime_features()
        export_path = self.analysis_dir / "regime_analysis.parquet"
        regime_df.to_parquet(export_path)

        return regime_analyzer

    def run_feature_analysis(self):
        print("\n" + "="*70)
        print("RUNNING FEATURE ANALYSIS")
        print("="*70)

        last_split = self.last_split
        analyser = FeatureAnalyser(
            model=last_split['model'],
            feature_names=self.feature_names,
            X_train=last_split['X_train'],
            y_train=last_split['y_train'],
            X_test=last_split['X_test'],
            y_test=last_split['y_test'],
        )

        analyser.print_importance_report(top_n=20)

        importance_df = analyser.get_tree_importance()
        importance_df.to_parquet(self.analysis_dir / "feature_importance.parquet")

        export = {
            "group_stats": analyser.analyse_feature_groups(),
            "feature_reduction": {
                "80pct": analyser.get_features_target_importance(0.80),
                "90pct": analyser.get_features_target_importance(0.90),
                "95pct": analyser.get_features_target_importance(0.95),
            },
            "redundant_pairs": [
                {"feature_a": a, "feature_b": b, "correlation": round(float(c), 4)}
                for a, b, c in analyser.find_redundant_features(correlation_threshold=0.95)
            ],
        }
        with open(self.analysis_dir / "feature_analysis.json", 'w') as f:
            json.dump(export, f, indent=2, default=float)

        return analyser

    def generate_visualizations(self):
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        import pandas as pd

        split_df = pd.DataFrame(self.split_results)
        shuffle_df = pd.DataFrame(self.shuffle_results)

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
        results['feature_analyser'] = self.run_feature_analysis()

        try:
            self.generate_visualizations()
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")

        return results

if __name__ == "__main__":
    config = load_config()
    print(config)
    orchestrator = AnalysisOrchestrator(config['run_dir'])
    orchestrator.run_full_analysis()