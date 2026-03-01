import streamlit as st
import pandas as pd
from utils.data_loader import load_all_experiment_metrics


def render():
    st.header("📊 Experiment Overview")

    with st.spinner("Loading experiments..."):
        df = load_all_experiment_metrics()

    if df.empty:
        st.warning("No experiments found. Run training first.")
        st.info("Expected directory structure: artifacts/experiment_name/run_id/")
        return

    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Experiments", df['experiment'].nunique())
    col2.metric("Total Runs", len(df))
    col3.metric("Avg IC", f"{df['ic_mean'].mean():.3f}" if 'ic_mean' in df.columns else "N/A")
    col4.metric("Avg Sharpe", f"{df['sharpe_ratio'].mean():.2f}" if 'sharpe_ratio' in df.columns else "N/A")

    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)

    with col1:
        min_ic = st.slider(
            "Minimum IC",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    with col2:
        min_sharpe = st.slider(
            "Minimum Sharpe",
            min_value=-2.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )

    if 'ic_mean' in df.columns:
        df = df[df['ic_mean'] >= min_ic]
    if 'sharpe_net' in df.columns:
        df = df[df['sharpe_net'] >= min_sharpe]

    st.subheader(f"Experiments ({len(df)} runs)")

    if df.empty:
        st.info("No experiments match the filters.")
        return

    display_df = df.copy()

    if 'experiment' in display_df.columns:
        display_df['experiment'] = display_df['experiment'].apply(lambda x: x.name if hasattr(x, 'name') else str(x))
    if 'run' in display_df.columns:
        display_df['run'] = display_df['run'].apply(lambda x: x.name if hasattr(x, 'name') else str(x))

    if 'ic_mean' in display_df.columns:
        display_df['ic_mean'] = display_df['ic_mean'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    if 'sharpe_net' in display_df.columns:
        display_df['sharpe_net'] = display_df['sharpe_net'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    if 'max_drawdown' in display_df.columns:
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    display_df = display_df.rename(columns={
        'experiment': 'Experiment',
        'run': 'Run ID',
        'ic_mean': 'Mean IC',
        'sharpe_ratio': 'Net Sharpe',
        'max_drawdown': 'Max Drawdown'
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    st.subheader("🏆 Best Performers")

    if 'sharpe_net' in df.columns and not df.empty:
        top_runs = df.nlargest(5, 'sharpe_net')

        for idx, row in top_runs.iterrows():
            exp_name = row['experiment'].name if hasattr(row['experiment'], 'name') else str(row['experiment'])
            run_name = row['run'].name if hasattr(row['run'], 'name') else str(row['run'])

            with st.expander(f"#{idx + 1}: {exp_name} - {run_name}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("IC", f"{row['ic_mean']:.3f}" if pd.notna(row.get('ic_mean')) else "N/A")
                col2.metric("Sharpe", f"{row['sharpe_net']:.2f}" if pd.notna(row.get('sharpe_net')) else "N/A")
                col3.metric("Drawdown", f"{row['max_drawdown']:.1%}" if pd.notna(row.get('max_drawdown')) else "N/A")

                if st.button(f"View Details", key=f"view_{idx}"):
                    st.session_state.selected_experiment = row['experiment']
                    st.session_state.selected_run = row['run']
                    st.session_state.page = "Backtest Results"
                    st.rerun()

    with st.expander("ℹ️ How to use this dashboard"):
        st.markdown("""
        **Overview Page (You are here):**
        - View summary of all experiments
        - Filter by IC or Sharpe ratio
        - Identify best performers

        **Experiment Details:**
        - Deep dive into validation metrics
        - View diagnostics and regime analysis

        **Backtest Results:**
        - View portfolio performance
        - Cumulative returns charts
        - Compare against benchmark

        **Strategy Comparison:**
        - Compare multiple strategies side-by-side
        - Overlaid performance charts
        """)