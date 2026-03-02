import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import (
    list_experiment, list_runs,
    load_backtest_results, load_backtest_returns,
    load_validation_results, get_run_metadata
)


def render():
    st.header("Strategy Comparison")

    all_runs = []
    experiments = list_experiment()

    for exp in experiments:
        for run in list_runs(exp):
            metadata = get_run_metadata(run)
            all_runs.append({
                'path': run,
                'display': f"{metadata['experiment_name']} - {run.name}",
                'experiment': metadata['experiment_name'],
                'run_id': run.name
            })

    if not all_runs:
        st.warning("No experiments found")
        return

    st.subheader("Select Strategies to Compare")

    selected_displays = st.multiselect(
        "Choose 2-5 strategies",
        [r['display'] for r in all_runs],
        default=[all_runs[0]['display']] if all_runs else []
    )

    if len(selected_displays) < 2:
        st.info("👆 Select at least 2 strategies to compare")
        return

    if len(selected_displays) > 5:
        st.warning("⚠️ Comparing more than 5 strategies can be cluttered. Consider selecting fewer.")

    selected_runs = [r for r in all_runs if r['display'] in selected_displays]

    comparison_data = []
    returns_data = {}

    with st.spinner("Loading data..."):
        for run_info in selected_runs:
            run_path = run_info['path']

            metrics = load_backtest_results(run_path)

            validation = load_validation_results(run_path)
            ic_mean = None
            if validation:
                rank_ic = validation.get('rank_ic', {})
                ic_mean = rank_ic.get('ic_mean') if isinstance(rank_ic, dict) else None

            returns = load_backtest_returns(run_path)

            if metrics:
                comparison_data.append({
                    'Strategy': run_info['display'],
                    'Sharpe Ratio': metrics.get('Sharpe Ratio'),
                    'Annual Return': metrics.get('Annual Return'),
                    'Annual Volatility': metrics.get('Annual Volatility'),
                    'Max Drawdown': metrics.get('Max Drawdown'),
                    'Mean IC': ic_mean,
                    'Turnover': metrics.get('Turnover (Annual)', 0)
                })

                if returns is not None:
                    returns_data[run_info['display']] = returns

    if not comparison_data:
        st.error("Could not load data for selected strategies")
        return

    comparison_df = pd.DataFrame(comparison_data)

    st.subheader("📈 Performance Metrics")

    display_df = comparison_df.copy()

    if 'Sharpe Ratio' in display_df.columns:
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    if 'Annual Return' in display_df.columns:
        display_df['Annual Return'] = display_df['Annual Return'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if 'Annual Volatility' in display_df.columns:
        display_df['Annual Volatility'] = display_df['Annual Volatility'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if 'Max Drawdown' in display_df.columns:
        display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if 'Win Rate' in display_df.columns:
        display_df['Win Rate'] = display_df['Win Rate'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if 'Mean IC' in display_df.columns:
        display_df['Mean IC'] = display_df['Mean IC'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    if 'Calmar Ratio' in display_df.columns:
        display_df['Calmar Ratio'] = display_df['Calmar Ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    if 'Turnover' in display_df.columns:
        display_df['Turnover'] = display_df['Turnover'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
        )

    st.dataframe(display_df, use_container_width=True)

    st.subheader("🔄 Delta Analysis")

    if 'Sharpe Ratio' in comparison_df.columns:
        best_idx = comparison_df['Sharpe Ratio'].idxmax()
        best_strategy = comparison_df.loc[best_idx, 'Strategy']

        st.info(f"**Best Strategy (by Sharpe):** {best_strategy}")

        delta_df = comparison_df.copy()

        for col in ['Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'Calmar Ratio', 'Mean IC']:
            if col in delta_df.columns:
                best_value = delta_df.loc[best_idx, col]
                delta_df[f'{col} Δ'] = delta_df[col] - best_value

        delta_display = delta_df[['Strategy']].copy()

        if 'Sharpe Ratio Δ' in delta_df.columns:
            delta_display['Sharpe Δ'] = delta_df['Sharpe Ratio Δ'].apply(
                lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A"
            )
        if 'Annual Return Δ' in delta_df.columns:
            delta_display['Return Δ'] = delta_df['Annual Return Δ'].apply(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
            )
        if 'Max Drawdown Δ' in delta_df.columns:
            delta_display['Drawdown Δ'] = delta_df['Max Drawdown Δ'].apply(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
            )

        st.dataframe(delta_display, use_container_width=True)

    st.subheader("📊 Cumulative Returns Comparison")

    if returns_data:
        fig = go.Figure()

        colors = ['#2D6A9F', '#1E6B3C', '#7A4F00', '#8B1A1A', '#6A1B9A']

        for idx, (strategy_name, returns_df) in enumerate(returns_data.items()):
            if 'Date' in returns_df.columns and 'Portfolio' in returns_df.columns:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(returns_df['Date']),
                    y=returns_df['Portfolio'],
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No returns data available for plotting")

    st.subheader("⚖️ Risk-Adjusted Performance")

    # Scatter plot: Return vs Volatility
    col1, col2 = st.columns(2)

    with col1:
        if 'Annual Return' in comparison_df.columns and 'Annual Volatility' in comparison_df.columns:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=comparison_df['Annual Volatility'],
                y=comparison_df['Annual Return'],
                mode='markers+text',
                text=comparison_df['Strategy'],
                textposition='top center',
                marker=dict(size=12, color='#2D6A9F'),
                showlegend=False
            ))

            fig.update_layout(
                xaxis_title="Annual Volatility",
                yaxis_title="Annual Return",
                height=400,
                title="Return vs Risk"
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Sharpe Ratio' in comparison_df.columns:
            fig = go.Figure(data=[
                go.Bar(
                    x=comparison_df['Strategy'],
                    y=comparison_df['Sharpe Ratio'],
                    marker_color=['#1E6B3C' if x == comparison_df['Sharpe Ratio'].max()
                                  else '#2D6A9F' for x in comparison_df['Sharpe Ratio']],
                    text=comparison_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}"),
                    textposition='outside'
                )
            ])

            fig.update_layout(
                yaxis_title="Sharpe Ratio",
                height=400,
                title="Sharpe Ratio Comparison",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("💾 Export")

    col1, col2 = st.columns(2)

    with col1:
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison (CSV)",
            data=csv,
            file_name="strategy_comparison.csv",
            mime="text/csv"
        )

    with col2:
        md = comparison_df.to_markdown(index=False)
        st.download_button(
            label="📄 Download Comparison (Markdown)",
            data=md,
            file_name="strategy_comparison.md",
            mime="text/markdown"
        )