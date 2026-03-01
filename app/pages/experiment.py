import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import (
    list_experiment, list_runs,
    load_validation_results,
    load_diagnostics_results,
    load_regime_analysis,
    load_latest_predictions  # NEW function
)


def render():
    st.header("Experiment Details")
    experiments = list_experiment()
    if not experiments:
        st.warning("No experiments found")
        return

    if 'selected_experiment' in st.session_state:
        default_exp = st.session_state.selected_experiment
        exp_idx = experiments.index(default_exp) if default_exp in experiments else 0
    else:
        exp_idx = 0

    experiment = st.sidebar.selectbox(
        "Experiment",
        experiments,
        index=exp_idx,
        format_func=lambda x: x.name if hasattr(x, 'name') else str(x)
    )
    runs = list_runs(experiment)
    if not runs:
        st.warning(f"No runs found for {experiment.name}")
        return

    if 'selected_run' in st.session_state:
        default_run = st.session_state.selected_run
        run_idx = runs.index(default_run) if default_run in runs else 0
    else:
        run_idx = 0

    run_id = st.sidebar.selectbox(
        "Run",
        runs,
        index=run_idx,
        format_func=lambda x: x.name if hasattr(x, 'name') else str(x)
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Validation",
        "Diagnostics",
        "Regime Analysis",
        "Live Predictions"
    ])

    with tab1:
        st.subheader("Statistical Validation")

        validation = load_validation_results(run_id)

        if not validation:
            st.warning("No validation results found")
        else:
            rank_ic = validation.get('rank_ic', {})

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean IC", f"{rank_ic.get('ic_mean', 0):.3f}")
            col2.metric("IC Std", f"{rank_ic.get('ic_std', 0):.3f}")
            col3.metric("Information Ratio", f"{rank_ic.get('information_ratio', 0):.2f}")
            col4.metric("P-value", f"{rank_ic.get('p_value', 0):.4f}")

            st.subheader("Statistical Tests")

            col1, col2 = st.columns(2)

            with col1:
                is_sig = rank_ic.get('p_value', 1) < 0.05
                st.metric(
                    "Significance Test",
                    "PASS ✅" if is_sig else "FAIL ❌",
                    f"p < 0.05" if is_sig else f"p = {rank_ic.get('p_value', 0):.4f}"
                )

            with col2:
                beats_shuffle = rank_ic.get('beats_random', False)
                st.metric(
                    "Beats Shuffle Test",
                    "PASS ✅" if beats_shuffle else "FAIL ❌",
                    "Signal > Random" if beats_shuffle else "Not significant"
                )

            st.subheader("IC Distribution Across Splits")

            st.json(validation)

    with tab2:
        st.subheader("Model Diagnostics")

        diagnostics = load_diagnostics_results(run_id)

        if not diagnostics:
            st.warning("No diagnostics results found")
        else:
            st.subheader("Stability Analysis")
            stability = diagnostics.get('stability', {})

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Coefficient of Variation",
                f"{stability.get('coefficient_of_variation', 0):.2f}"
            )
            col2.metric(
                "IQR",
                f"{stability.get('interquartile_range', 0):.3f}"
            )
            col3.metric(
                "Outlier Splits",
                f"{len(stability.get('outlier_splits', []))}"
            )

            if 'category_distribution' in stability:
                st.subheader("Performance Categories")
                cat_dist = stability['category_distribution']

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(cat_dist.keys()),
                        y=list(cat_dist.values()),
                        marker_color=['#8B1A1A', '#7A4F00', '#2D6A9F', '#1E6B3C']
                    )
                ])
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Number of Splits",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Temporal Trends")
            temporal = diagnostics.get('temporal', {})

            col1, col2, col3 = st.columns(3)
            col1.metric("Trend Slope", f"{temporal.get('trend_slope', 0):.6f}")
            col2.metric("Trend P-value", f"{temporal.get('trend_pvalue', 0):.4f}")
            col3.metric("Autocorrelation", f"{temporal.get('autocorrelation', 0):.3f}")

            with st.expander("Full Diagnostics Report"):
                st.json(diagnostics)

    with tab3:
        st.subheader("Regime Dependency Analysis")

        regime_df = load_regime_analysis(run_id)

        if regime_df is None or regime_df.empty:
            st.warning("No regime analysis found")
        else:
            st.write(f"Analyzed {len(regime_df)} test periods")

            st.subheader("Market Features vs IC")

            feature_cols = [col for col in regime_df.columns
                            if col not in ['split', 'rank_ic', 'test_start', 'test_end']]

            if 'rank_ic' in regime_df.columns and feature_cols:
                correlations = regime_df[feature_cols + ['rank_ic']].corr()['rank_ic'].drop('rank_ic')
                correlations = correlations.sort_values(ascending=False)

                fig = go.Figure(data=[
                    go.Bar(
                        x=correlations.values,
                        y=correlations.index,
                        orientation='h',
                        marker_color=['#1E6B3C' if x > 0 else '#8B1A1A' for x in correlations.values]
                    )
                ])
                fig.update_layout(
                    xaxis_title="Correlation with IC",
                    yaxis_title="Market Feature",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Good vs Bad Regime Characteristics")

            if 'rank_ic' in regime_df.columns:
                threshold = st.slider("IC Threshold", 0.0, 0.5, 0.25, 0.05)

                good_regimes = regime_df[regime_df['rank_ic'] > threshold]
                bad_regimes = regime_df[regime_df['rank_ic'] <= threshold]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Good Regime Periods", len(good_regimes))
                    if not good_regimes.empty and feature_cols:
                        st.write("Average Characteristics:")
                        for col in feature_cols[:5]:  # Show top 5
                            st.write(f"• {col}: {good_regimes[col].mean():.3f}")

                with col2:
                    st.metric("Bad Regime Periods", len(bad_regimes))
                    if not bad_regimes.empty and feature_cols:
                        st.write("Average Characteristics:")
                        for col in feature_cols[:5]:
                            st.write(f"• {col}: {bad_regimes[col].mean():.3f}")

            with st.expander("Full Regime Data"):
                st.dataframe(regime_df)


    with tab4:
        st.subheader("🎯 Current Model Predictions")

        predictions = load_latest_predictions(run_id)

        if predictions is None or predictions.empty:
            st.warning("No predictions found for this run.")
            st.info("Predictions are saved during training in: predictions/live_signal.csv")
        else:
            latest_date = predictions['Date'].max()
            latest_preds = predictions[predictions['Date'] == latest_date].copy()

            st.write(f"**Prediction Date:** {latest_date}")
            st.write(f"**Number of Stocks:** {len(latest_preds)}")

            latest_preds = latest_preds.sort_values('signal', ascending=False)

            st.subheader("🟢 Top Long Positions (Top 20%)")
            top_20_pct = int(len(latest_preds) * 0.2)
            top_longs = latest_preds.head(top_20_pct)

            display_longs = top_longs[['Ticker', 'signal']].copy()
            display_longs['signal'] = display_longs['signal'].apply(lambda x: f"{x:.4f}")
            display_longs = display_longs.rename(columns={'signal': 'Signal Strength'})

            st.dataframe(display_longs, use_container_width=True, height=300)

            st.subheader("🔴 Top Short Positions (Bottom 20%)")
            bottom_20_pct = int(len(latest_preds) * 0.2)
            top_shorts = latest_preds.tail(bottom_20_pct)

            display_shorts = top_shorts[['Ticker', 'signal']].copy()
            display_shorts['signal'] = display_shorts['signal'].apply(lambda x: f"{x:.4f}")
            display_shorts = display_shorts.rename(columns={'signal': 'Signal Strength'})

            st.dataframe(display_shorts, use_container_width=True, height=300)

            st.subheader("Signal Distribution")

            fig = go.Figure(data=[
                go.Histogram(
                    x=latest_preds['signal'],
                    nbinsx=30,
                    marker_color='#2D6A9F'
                )
            ])
            fig.update_layout(
                xaxis_title="Prediction",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = latest_preds.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{latest_date}.csv",
                mime="text/csv"
            )

            with st.expander("View All Predictions"):
                st.dataframe(latest_preds)