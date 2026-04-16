from backtest.align_data import DataAligner
from backtest.data_loaders import load_backtest_data
from backtest.portfolio import PortfolioConstructor
from backtest.simple_backtester import SimpleBacktest
from config.loader import load_config
import mlflow
import os
import boto3

class BacktestOrchestrator:
    def __init__(self, config):
        self.run_id = config['run_id']
        self.experiment = config['experiment_name']
        self.horizon = config['horizon']
        self.portfolio_strategy = config['strategy']
        self.benchmark = config['benchmark']

        self.tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.run_name = config['run_id']
        self.experiment_name = config.get('experiment_name', 'default_experiment')

        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.set_experiment(self.experiment_name)

        existing = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'",
            max_results=1,
        )
        if len(existing) == 0:
            raise RuntimeError(
                f"No MLflow run named '{self.run_name}' found — "
                f"did training complete?"
            )
        self.mlflow_run_id = existing.iloc[0]["run_id"]

    def run(self):
        data = load_backtest_data(self.experiment, self.run_id, self.horizon, self.benchmark)
        aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
        combined_data, close_prices = aligner.align()

        constructor = PortfolioConstructor(strategy=self.portfolio_strategy)
        portfolio = constructor.construct(combined_data)

        backtester = SimpleBacktest(portfolio, close_prices, horizon_months=self.horizon, benchmark=self.benchmark)

        tes = backtester.run()
        tes.save(f"../training/artifacts/{self.experiment}/{self.run_id}")
        tes.print_summary()

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_param("backtest_horizon", self.horizon)
            mlflow.log_param("backtest_strategy", self.portfolio_strategy)
            mlflow.log_param("backtest_benchmark", self.benchmark)

            tes = backtester.run()

            save_path = f"../training/artifacts/{self.experiment}/{self.run_id}"
            tes.save(save_path)
            tes.print_summary()

            metrics = {
                "backtest_sharpe": tes.sharpe,
                "backtest_total_return": tes.total_return,
                "backtest_max_drawdown": tes.max_drawdown,
                "backtest_volatility": tes.volatility,
                "backtest_calmar": tes.calmar,
                "backtest_win_rate": tes.win_rate,
            }
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

            if hasattr(tes, "alpha"):
                mlflow.log_metric("backtest_alpha", tes.alpha)
                mlflow.log_metric("backtest_beats_benchmark", int(tes.sharpe > tes.benchmark_sharpe))

            mlflow.log_artifacts(save_path, artifact_path="backtest")

            mlflow.set_tag("backtest_profitable", str(tes.total_return > 0))
            mlflow.set_tag("backtest_strategy", self.portfolio_strategy)

if __name__ == "__main__":
    config = load_config()
    data_pipeline = BacktestOrchestrator(config)
    data_pipeline.run()