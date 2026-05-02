"""Streamlit page functions for the project dashboard."""

from typing import Any

from src.experiments.configs import SUPPORTED_METHODS, SUPPORTED_SCALES
from src.experiments.configs import get_default_config
from src.experiments.paths import PLOTS_DIR, RAW_RESULTS_DIR
from src.experiments.results import load_results
from src.vis.tables import make_summary_table


def _require_streamlit() -> Any:
    import streamlit as st

    return st


def render_dataset_manager() -> None:
    """Render dataset metadata and project split information."""
    st = _require_streamlit()

    st.header("Dataset Manager")
    st.write("Dataset: `ogbl-collab`")
    st.write("Task: link prediction on academic collaboration edges")
    st.write("Primary metric: `Hits@50`")
    st.write("Planned dataset scales: `10%`, `50%`, `100%`")
    st.info(
        "Dataset loading is performed through the installed `ogb` package. "
        "The local `ogb/` source tree is not imported by the project code."
    )


def render_algorithm_runner() -> None:
    """Render a lightweight command builder for experiment runs."""
    st = _require_streamlit()

    st.header("Algorithm Runner")
    method_name = st.selectbox("Method", SUPPORTED_METHODS)
    dataset_scale = st.selectbox("Dataset scale", SUPPORTED_SCALES)
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    device = st.selectbox("Device", ("cpu", "cuda:0"))

    config = get_default_config(
        method_name=method_name,
        dataset_scale=float(dataset_scale),
        seed=int(seed),
        device=device,
    )

    st.subheader("CLI command")
    st.code(
        "python src/train.py "
        f"--method {config.method_name} "
        f"--scale {config.dataset_scale} "
        f"--seed {config.seed} "
        f"--device {config.device}",
        language="bash",
    )
    st.caption("Run experiments from the terminal for reliable long-running execution.")


def render_results_dashboard() -> None:
    """Render saved metrics and generated plot paths."""
    st = _require_streamlit()

    st.header("Results Dashboard")
    results = load_results(RAW_RESULTS_DIR)

    if not results:
        st.warning("No saved result JSON files found under `results/raw/`.")
        return

    rows = make_summary_table(results)
    st.subheader("Summary Table")
    st.dataframe(rows, use_container_width=True)

    st.subheader("Generated Plots")
    for plot_path in sorted(PLOTS_DIR.glob("*.png")):
        st.image(str(plot_path), caption=plot_path.name)
