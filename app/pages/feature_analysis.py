import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json

from utils.data_loader import list_experiment, list_runs, load_feature_importance, load_feature_analysis


def render():

    st.header("Feature Analysis")

    experiments = list_experiment()
    if not experiments:
        st.warning("No experiments found.")
        return

    experiment = st.sidebar.selectbox(
        "Experiment",
        experiments,
        format_func=lambda x: x.name
    )

    runs = list_runs(experiment)
    if not runs:
        st.warning("No runs found for this experiment.")
        return

    run_id = st.sidebar.selectbox(
        "Run",
        runs,
        format_func=lambda x: x.name
    )

    importance_df = load_feature_importance(run_id)
    analysis_json = load_feature_analysis(run_id)

    if importance_df is None:
        st.warning("Feature importance file not found.")
        return

    if analysis_json is None:
        st.warning("Feature analysis JSON not found.")
        return

    st.subheader("Tree-Based Feature Importance")

    importance_df = importance_df.sort_values("importance", ascending=False)

    top_n = st.slider("Top N Features", 10, 100, 30, 5)
    top_features = importance_df.head(top_n)

    fig = go.Figure(data=[
        go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h"
        )
    ])
    fig.update_layout(
        height=600,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top_features, use_container_width=True)

    st.subheader("Feature Group Analysis")

    group_stats = analysis_json.get("group_stats", {})
    if group_stats:
        group_df = pd.DataFrame(group_stats).T
        group_df = group_df.sort_values("total_importance", ascending=False)

        fig2 = go.Figure(data=[
            go.Bar(
                x=group_df["total_importance"],
                y=group_df.index,
                orientation="h"
            )
        ])
        fig2.update_layout(
            height=400,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Total Group Importance",
            yaxis_title="Feature Group"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(group_df, use_container_width=True)

    st.subheader("Feature Reduction Opportunities")

    reduction = analysis_json.get("feature_reduction", {})
    total_features = len(importance_df)

    for pct_label, features in reduction.items():
        st.write(
            f"{pct_label} importance: {len(features)} features "
            f"({len(features)/total_features:.1%} of total)"
        )
        st.write(features)

    st.subheader("Highly Correlated Features")

    redundant = analysis_json.get("redundant_pairs", [])
    if redundant:
        st.write(f"Found {len(redundant)} pairs of highly correlated features:")
        st.dataframe(pd.DataFrame(redundant).rename(columns={
            "feature_a": "Feature A",
            "feature_b": "Feature B",
            "correlation": "Correlation"
        }), use_container_width=True)
    else:
        st.write("No highly correlated features found.")