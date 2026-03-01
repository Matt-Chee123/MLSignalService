import streamlit as st
import pandas as pd
from pathlib import Path
import json

ARTIFACTS_DIR = Path('../training/artifacts')


def load_experiment_summary():
    rows = []
    if not ARTIFACTS_DIR.exists():
        return pd.DataFrame()

    for experiment_dir in ARTIFACTS_DIR.iterdir():
        if not experiment_dir.is_dir():
            continue

        best_sharpe = None
        sharpe_list = []
        ic_list = []

        for run_dir in experiment_dir.iterdir():
            metrics_path = run_dir / "backtest" / "metrics.json"
            validation_path = run_dir / "analysis" / "validation_results.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                sharpe = metrics.get("Sharpe Ratio")
                if sharpe is not None:
                    sharpe_list.append(sharpe)
                    if best_sharpe is None or sharpe > best_sharpe:
                        best_sharpe = sharpe

            if validation_path.exists():
                with open(validation_path) as f:
                    val = json.load(f)
                ic = val.get("rank_ic", {}).get("ic_mean")
                if ic is not None:
                    ic_list.append(ic)

        if sharpe_list:
            rows.append({
                "Experiment": experiment_dir.name,
                "Runs": len(sharpe_list),
                "Best Sharpe": best_sharpe,
                "Avg Sharpe": sum(sharpe_list) / len(sharpe_list),
                "Avg IC": sum(ic_list) / len(ic_list) if ic_list else 0.0
            })

    return pd.DataFrame(rows)


def render():
    st.header("Overview")
    df = load_experiment_summary()

    if df.empty:
        st.warning("No experiments found.")
        return

    st.sidebar.subheader("Filters")

    df["Avg IC"] = df["Avg IC"].fillna(0.0)

    ic_min = float(df["Avg IC"].min())
    ic_max = float(df["Avg IC"].max())

    if ic_min == ic_max:
        min_ic = ic_min
        st.sidebar.text(f"Minimum Avg IC: {min_ic:.4f} (all experiments have same IC)")
    else:
        min_ic = st.sidebar.slider(
            "Minimum Avg IC",
            min_value=ic_min,
            max_value=ic_max,
            value=ic_min
        )

    df_filtered = df[df["Avg IC"] >= min_ic]

    st.subheader("Experiment Summary")
    st.dataframe(
        df_filtered.sort_values("Best Sharpe", ascending=False),
        use_container_width=True
    )

    st.subheader("Global Stats")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Experiments", len(df))
    col2.metric("Average Sharpe", round(df["Avg Sharpe"].mean(), 3))
    col3.metric("Average IC", round(df["Avg IC"].mean(), 4))