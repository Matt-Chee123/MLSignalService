from backtest.data_loaders import load_backtest_data
from config.backtest_config import HORIZON


class DataAligner:
    def __init__(self, predictions, prices, returns):
        self.predictions = predictions
        self.prices = prices
        self.returns = returns

        self.aligned_data = None
        self.coverage_report = None

    def align(self):
        combined = self.predictions.join(
            self.returns[['forward_return']],
            how='left'
        )
        self.compute_coverage(combined)

        initial_count = len(combined)

        combined = combined.dropna(subset=['pred_label'])
        after_pred_drop = len(combined)
        pred_dropped = initial_count - after_pred_drop

        combined = combined.dropna(subset=['forward_return'])
        after_return_drop = len(combined)
        return_dropped = after_pred_drop - after_return_drop

        combined = combined.sort_index()

        assert combined['pred_label'].isna().sum() == 0
        assert combined['forward_return'].isna().sum() == 0

        self.aligned_data = combined

        self.coverage_report['predictions_dropped'] = pred_dropped
        self.coverage_report['returns_dropped'] = return_dropped
        self.coverage_report['final_count'] = len(combined)

        final_df = combined[['pred_label','forward_return','Close']]

        return final_df

    def compute_coverage(self, df):
        total = len(df)

        missing_labels = df['pred_label'].isna().sum()
        missing_return = df['forward_return'].isna().sum()
        missing_close = df['Close'].isna().sum()
        dates_total = df.index.get_level_values('Date').nunique()
        dates_with_full_coverage = (
            df.groupby(level='Date')
            .apply(lambda x: x['forward_return'].notna().all())
            .sum()
        )
        self.coverage_report = {
            'total_predictions': total,
            'missing_predictions': missing_labels,
            'missing_returns': missing_return,
            'missing_prices': missing_close,
            'coverage_pct': 100 * (total - missing_return) / total if total > 0 else 0,
            'dates_total': dates_total,
            'dates_full_coverage': dates_with_full_coverage,
            'dates_partial_coverage': dates_total - dates_with_full_coverage
        }

    def get_coverage_report(self):
        if self.coverage_report is None:
            raise ValueError("Must call align() before getting coverage report")

        return self.coverage_report

    def print_coverage_report(self):
        if self.coverage_report is None:
            raise ValueError("Must call align() before printing coverage report")

        r = self.coverage_report

        print("\n" + "=" * 60)
        print("DATA ALIGNMENT COVERAGE REPORT")
        print("=" * 60)
        print(f"Total predictions:           {r['total_predictions']:,}")
        print(f"Final aligned observations:  {r['final_count']:,}")
        print(f"Coverage:                    {r['coverage_pct']:.1f}%")
        print()
        print(f"Missing predictions:         {r['missing_predictions']:,}")
        print(f"Missing forward returns:     {r['missing_returns']:,}")
        print(f"Missing prices:              {r['missing_prices']:,}")
        print()
        print(f"Predictions dropped:         {r['predictions_dropped']:,}")
        print(f"Returns dropped:             {r['returns_dropped']:,}")
        print()
        print(f"Dates with full coverage:    {r['dates_full_coverage']} / {r['dates_total']}")
        print(f"Dates with partial coverage: {r['dates_partial_coverage']}")
        print("=" * 60 + "\n")

data = load_backtest_data('rf_signal_v1', '20260219_210026', HORIZON)
aligner = DataAligner(data['predictions'],data['prices'],data['returns'])
aligner.align()