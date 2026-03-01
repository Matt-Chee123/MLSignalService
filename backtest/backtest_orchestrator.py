from align_data import DataAligner
from data_loaders import load_backtest_data
from portfolio import PortfolioConstructor
from simple_backtester import SimpleBacktest


class BacktestOrchestrator:
    def __init__(self, config):
        self.run_id = config['run_id']
        self.experiment = config['experiment']
        self.horizon = config['horizon']
        self.portfolio_strategy = config['strategy']
        self.benchmark = config['benchmark']

    def run(self):
        data = load_backtest_data(self.experiment, self.run_id, self.horizon)
        aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
        combined_data, close_prices = aligner.align()

        constructor = PortfolioConstructor(strategy=self.portfolio_strategy)
        portfolio = constructor.construct(combined_data)

        backtester = SimpleBacktest(portfolio, close_prices, horizon_months=self.horizon, benchmark=self.benchmark)

        tes = backtester.run()
        tes.save(f"../training/artifacts/{self.experiment}/{self.run_id}")
