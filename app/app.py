import streamlit as st
from pages import overview, experiment, backtest, comparison, raw_data, feature_analysis

def main():
    st.title("ML Signal Experiment Dashboard")

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    page = st.sidebar.selectbox(
        'Navigate',
        ['Overview', 'Experiment Details', 'Backtest Results', 'Strategy Comparison', 'Feature Analysis', 'Raw Data'],
        index=['Overview', 'Experiment Details', 'Backtest Results', 'Strategy Comparison', 'Feature Analysis', 'Raw Data']
        .index(st.session_state.page)
    )
    st.session_state.page = page
    if page == 'Overview':
        overview.render()
    elif page == 'Experiment Details':
        experiment.render()
    elif page == 'Backtest Results':
        backtest.render()
    elif page == 'Strategy Comparison':
        comparison.render()
    elif page == 'Feature Analysis':
        feature_analysis.render()
    else:
        raw_data.render()


if __name__ == "__main__":
    main()