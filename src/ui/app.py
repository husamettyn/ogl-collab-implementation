"""Streamlit dashboard application."""

from src.ui.pages import render_algorithm_runner
from src.ui.pages import render_graph_explorer
from src.ui.pages import render_results_dashboard


def main() -> None:
    """Run the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="ogbl-collab Link Prediction",
        layout="wide",
    )
    st.title("ogbl-collab Link Prediction")

    page = st.sidebar.radio(
        "Page",
        (
            "Graph Explorer",
            "Algorithm Runner",
            "Results Dashboard",
        ),
    )

    if page == "Graph Explorer":
        render_graph_explorer()
    elif page == "Algorithm Runner":
        render_algorithm_runner()
    else:
        render_results_dashboard()


if __name__ == "__main__":
    main()
