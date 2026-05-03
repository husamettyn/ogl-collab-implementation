"""Results dashboard page."""

from __future__ import annotations

from collections import Counter

from src.experiments.paths import PLOTS_DIR, RAW_RESULTS_DIR
from src.experiments.results import load_results
from src.ui.common import require_streamlit
from src.vis.tables import make_best_results_table, make_summary_table


def render_results_dashboard() -> None:
    """Render saved metrics and generated plot paths."""
    st = require_streamlit()

    st.header("Results Dashboard")
    results = load_results(RAW_RESULTS_DIR)
    if not results:
        st.warning("No saved result JSON files found under `results/raw/`.")
        return

    all_rows = make_summary_table(results)
    methods = sorted({str(result.get("method_name")) for result in results if result.get("method_name")})
    scales = sorted({float(result.get("dataset_scale")) for result in results if result.get("dataset_scale") is not None})
    statuses = sorted({str(result.get("status")) for result in results if result.get("status")})
    splits = sorted({str(row.get("split")) for row in all_rows if row.get("split")})
    seeds = sorted({int(result.get("seed")) for result in results if result.get("seed") is not None})

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    selected_methods = filter_col1.multiselect("Methods", methods, default=methods)
    selected_splits = filter_col2.multiselect("Splits", splits, default=splits)
    selected_statuses = filter_col3.multiselect("Statuses", statuses, default=statuses)

    filter_col4, filter_col5 = st.columns(2)
    selected_scales = filter_col4.multiselect("Scales", scales, default=scales)
    selected_seeds = filter_col5.multiselect("Seeds", seeds, default=seeds)

    filtered_results = [
        result
        for result in results
        if result.get("method_name") in selected_methods
        and result.get("status") in selected_statuses
        and float(result.get("dataset_scale")) in selected_scales
        and int(result.get("seed")) in selected_seeds
    ]
    filtered_rows = [
        row
        for row in make_summary_table(filtered_results)
        if row.get("split") in selected_splits
    ]

    if not filtered_rows:
        st.warning("No rows match current filters.")
        return

    test_rows = [row for row in filtered_rows if row.get("split") == "test"]
    kpi_rows = test_rows if test_rows else filtered_rows
    best_hits = max(
        (float(row["hits_at_50"]) for row in kpi_rows if row.get("hits_at_50") is not None),
        default=0.0,
    )
    fastest_runtime = min(
        (float(result["runtime_seconds"]) for result in filtered_results if result.get("runtime_seconds") is not None),
        default=0.0,
    )
    min_memory = min(
        (float(result["memory_mb"]) for result in filtered_results if result.get("memory_mb") is not None),
        default=0.0,
    )
    split_counter = Counter(str(row.get("split")) for row in filtered_rows if row.get("split"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Best Hits@50", f"{best_hits:.4f}")
    metric_cols[1].metric("Fastest Runtime (s)", f"{fastest_runtime:.2f}")
    metric_cols[2].metric("Lowest Memory (MB)", f"{min_memory:.2f}")
    metric_cols[3].metric("Visible Rows", len(filtered_rows))
    st.caption(f"Split row counts: {dict(split_counter)}")

    st.subheader("Summary Table")
    st.dataframe(filtered_rows, width="stretch")

    st.subheader("Best Rows by Method and Scale")
    preferred_split = selected_splits[0] if selected_splits else "test"
    best_rows = make_best_results_table(filtered_results, split=preferred_split)
    st.dataframe(best_rows, width="stretch")

    runs_with_losses = [result for result in filtered_results if result.get("losses")]
    if runs_with_losses:
        st.subheader("Loss Curve")
        run_labels = [
            f"{result.get('timestamp')} | {result.get('method_name')} | scale={result.get('dataset_scale')}"
            for result in runs_with_losses
        ]
        selected_label = st.selectbox("Run", run_labels)
        selected_result = runs_with_losses[run_labels.index(selected_label)]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.5))
        losses = [float(value) for value in selected_result["losses"]]
        ax.plot(range(1, len(losses) + 1), losses, color="#4e79a7")
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    st.subheader("Generated Plots")
    for plot_path in sorted(PLOTS_DIR.glob("*.png")):
        st.image(str(plot_path), caption=plot_path.name)
