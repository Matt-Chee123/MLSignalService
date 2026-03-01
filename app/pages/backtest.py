import streamlit as st
import plotly.graph_objs as go
from utils.data_loader import list_runs, list_experiment, load_backtest_results, load_backtest_returns
import pandas as pd

def render():
    st.header("Backtest Results")

    experiments = list_experiment()
    if not experiments:
        st.warning("No experiments found")
        return

    experiment = st.sidebar.selectbox("Experiment", experiments)
    runs = list_runs(experiment)
    run_id = st.sidebar.selectbox("Run", runs)

    metrics = load_backtest_results(run_id)
    returns = load_backtest_returns(run_id)

    if not metrics:
        st.warning("No backtest results")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio (net)", f"{metrics['Sharpe Ratio']:.2f}")
    col2.metric("Annual Return", f"{metrics['Annual Return']:.1%}")
    col3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")
    col4.metric("Mean Period Return", f"{metrics['Mean Period Return']:.1%}")

    st.subheader("Cumulative Returns")
    if returns is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(returns['Date']),
            y=returns['Portfolio'],
            mode="lines",
            name="Strategy",
            line=dict(color="#2D6A9F", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(returns['Date']),
            y=returns['Benchmark'],
            mode="lines",
            name="Benchmark",
            line=dict(color="#FF5733", width=2)
        ))

        fig.update_layout(
            xaxis=dict(title="Date"),
            yaxis=dict(title="Cumulative Return"),
            height=400,
            legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Metrics")
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ["Value"]
    st.dataframe(metrics_df)