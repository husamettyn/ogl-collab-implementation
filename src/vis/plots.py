"""Plotting helpers for saved experiment results."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.experiments.paths import PLOTS_DIR, ensure_result_dirs
from src.vis.tables import make_summary_table


def _filter_rows(
    results: Iterable[dict[str, Any]],
    split: str,
    value_key: str,
) -> list[dict[str, Any]]:
    rows = make_summary_table(results)
    return [
        row
        for row in rows
        if row.get("split") == split and row.get(value_key) is not None
    ]


def _plot_bar_rows(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    output_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    ensure_result_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        f"{row['method_name']}\nscale={row['dataset_scale']}"
        for row in rows
    ]
    values = [float(row[value_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(value_key)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_hits_comparison(
    results: Iterable[dict[str, Any]],
    output_path: Path | str = PLOTS_DIR / "hits_at_50_comparison.png",
    metric_key: str = "hits_at_50",
    split: str = "test",
) -> Path:
    """Save a Hits@K comparison chart and return its path."""
    rows = _filter_rows(results=results, split=split, value_key=metric_key)
    return _plot_bar_rows(
        rows=rows,
        value_key=metric_key,
        title=f"{metric_key} comparison on {split}",
        output_path=Path(output_path),
    )


def plot_runtime_comparison(
    results: Iterable[dict[str, Any]],
    output_path: Path | str = PLOTS_DIR / "runtime_comparison.png",
) -> Path:
    """Save a runtime comparison chart and return its path."""
    rows = [
        result
        for result in results
        if result.get("runtime_seconds") is not None
    ]
    return _plot_bar_rows(
        rows=rows,
        value_key="runtime_seconds",
        title="Runtime comparison",
        output_path=Path(output_path),
    )


def plot_memory_comparison(
    results: Iterable[dict[str, Any]],
    output_path: Path | str = PLOTS_DIR / "memory_comparison.png",
) -> Path:
    """Save a memory comparison chart and return its path."""
    rows = [
        result
        for result in results
        if result.get("memory_mb") is not None
    ]
    return _plot_bar_rows(
        rows=rows,
        value_key="memory_mb",
        title="Memory usage comparison",
        output_path=Path(output_path),
    )


def save_all_plots(results: Iterable[dict[str, Any]]) -> dict[str, Path]:
    """Generate the standard project plot set."""
    results = list(results)
    return {
        "hits_at_50": plot_hits_comparison(results),
        "runtime": plot_runtime_comparison(results),
        "memory": plot_memory_comparison(results),
    }
