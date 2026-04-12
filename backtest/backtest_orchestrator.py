from backtest.align_data import DataAligner
from backtest.data_loaders import load_backtest_data
from backtest.portfolio import PortfolioConstructor
from backtest.simple_backtester import SimpleBacktest
from config.loader import load_config


class BacktestOrchestrator:
    def __init__(self, config):
        self.run_id = config['run_id']
        self.experiment = config['experiment_name']
        self.horizon = config['horizon']
        self.portfolio_strategy = config['strategy']
        self.benchmark = config['benchmark']

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

if __name__ == "__main__":
    config = load_config()
    data_pipeline = BacktestOrchestrator(config)
    data_pipeline.run()