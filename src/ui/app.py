"""Streamlit dashboard application."""

from src.ui.pages import render_algorithm_runner
from src.ui.pages import render_dataset_manager
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
        ("Dataset Manager", "Algorithm Runner", "Results Dashboard"),
    )

    if page == "Dataset Manager":
        render_dataset_manager()
    elif page == "Algorithm Runner":
        render_algorithm_runner()
    else:
        render_results_dashboard()


if __name__ == "__main__":
    main()
