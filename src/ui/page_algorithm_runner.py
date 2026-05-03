"""Algorithm runner page."""

from __future__ import annotations

from typing import Any

from src.experiments.configs import (
    SUPPORTED_METHODS,
    SUPPORTED_SCALES,
    ExperimentConfig,
    get_method_config,
)
from src.experiments.runner import run_experiment
from src.experiments.runtime_config import configure_logging, suppress_known_warnings
from src.ui.common import RUNNER_PRESETS, require_streamlit


def _command_for_runner(
    method_name: str,
    dataset_scale: float,
    seed: int,
    device: str,
    hyperparameters: dict[str, Any],
    save_result: bool,
) -> str:
    parts = [
        "python src/train.py",
        f"--method {method_name}",
        f"--scale {dataset_scale}",
        f"--seed {seed}",
        f"--device {device}",
    ]
    if method_name == "common_neighbors":
        if not hyperparameters.get("add_tie_breaker", True):
            parts.append("--disable-tie-breaker")
        if not hyperparameters.get("make_undirected", True):
            parts.append("--directed")
    else:
        parts.extend(
            [
                f"--epochs {hyperparameters['epochs']}",
                f"--batch-size {hyperparameters['batch_size']}",
                f"--hidden-channels {hyperparameters['hidden_channels']}",
                f"--num-layers {hyperparameters['num_layers']}",
                f"--dropout {hyperparameters['dropout']}",
                f"--learning-rate {hyperparameters['learning_rate']}",
            ]
        )
    if not save_result:
        parts.append("--no-save")
    return " ".join(parts)


def render_algorithm_runner() -> None:
    """Render a command builder for experiment runs."""
    st = require_streamlit()

    st.header("Algorithm Runner")

    with st.form("algorithm_runner_form"):
        col1, col2, col3 = st.columns(3)
        method_name = col1.selectbox("Method", SUPPORTED_METHODS)
        dataset_scale = float(col2.selectbox("Dataset scale", SUPPORTED_SCALES))
        preset = col3.selectbox("Preset", tuple(RUNNER_PRESETS))

        col4, col5 = st.columns(2)
        seed = int(col4.number_input("Seed", min_value=0, value=42, step=1))
        device = col5.selectbox("Device", ("cpu", "cuda", "cuda:0"))
        save_result = st.checkbox("Save result JSON under results/raw/", value=True)

        hyperparameters = get_method_config(method_name)
        hyperparameters.update(RUNNER_PRESETS[preset])

        if method_name == "common_neighbors":
            tie_breaker = st.checkbox(
                "Use deterministic tie breaker",
                value=bool(hyperparameters.get("add_tie_breaker", True)),
            )
            undirected = st.checkbox(
                "Treat graph as undirected",
                value=bool(hyperparameters.get("make_undirected", True)),
            )
            hyperparameters["add_tie_breaker"] = tie_breaker
            hyperparameters["make_undirected"] = undirected
        else:
            pcol1, pcol2, pcol3 = st.columns(3)
            hyperparameters["epochs"] = int(
                pcol1.number_input(
                    "Epochs",
                    min_value=1,
                    value=int(hyperparameters["epochs"]),
                    step=1,
                )
            )
            hyperparameters["batch_size"] = int(
                pcol2.number_input(
                    "Batch size",
                    min_value=32,
                    value=int(hyperparameters["batch_size"]),
                    step=32,
                )
            )
            hyperparameters["hidden_channels"] = int(
                pcol3.number_input(
                    "Hidden channels",
                    min_value=16,
                    value=int(hyperparameters["hidden_channels"]),
                    step=16,
                )
            )

            pcol4, pcol5, pcol6 = st.columns(3)
            hyperparameters["num_layers"] = int(
                pcol4.number_input(
                    "Num layers",
                    min_value=1,
                    value=int(hyperparameters["num_layers"]),
                    step=1,
                )
            )
            hyperparameters["dropout"] = float(
                pcol5.number_input(
                    "Dropout",
                    min_value=0.0,
                    max_value=0.9,
                    value=float(hyperparameters["dropout"]),
                    step=0.05,
                    format="%.2f",
                )
            )
            hyperparameters["learning_rate"] = float(
                pcol6.number_input(
                    "Learning rate",
                    min_value=0.00001,
                    max_value=1.0,
                    value=float(hyperparameters["learning_rate"]),
                    step=0.0005,
                    format="%.5f",
                )
            )

        run_requested = st.form_submit_button("Run Experiment", type="primary")

    train_command = _command_for_runner(
        method_name=method_name,
        dataset_scale=dataset_scale,
        seed=seed,
        device=device,
        hyperparameters=hyperparameters,
        save_result=save_result,
    )
    root_command = train_command.replace("python src/train.py", "python main.py train", 1)

    if run_requested:
        config = ExperimentConfig(
            method_name=method_name,
            dataset_scale=dataset_scale,
            seed=seed,
            device=device,
            save_result=save_result,
            hyperparameters=hyperparameters,
        )
        configure_logging()
        suppress_known_warnings()

        with st.spinner("Experiment calisiyor... Bu islem methoda gore zaman alabilir."):
            try:
                result = run_experiment(config)
            except Exception as error:
                st.error(f"Run basarisiz: {error}")
                st.exception(error)
                return

        st.success("Run tamamlandi.")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        status_col1.metric("Method", str(result.get("method_name", "-")))
        status_col2.metric("Scale", str(result.get("dataset_scale", "-")))
        status_col3.metric("Status", str(result.get("status", "-")))
        runtime_value = result.get("runtime_seconds")
        status_col4.metric(
            "Runtime (s)",
            f"{float(runtime_value):.2f}" if runtime_value is not None else "-",
        )

        if result.get("metrics"):
            st.subheader("Run Metrics")
            st.json(result["metrics"])

        if result.get("result_path"):
            st.caption(f"Saved result: {result['result_path']}")

    with st.expander("CLI commands", expanded=False):
        st.subheader("CLI command")
        st.code(train_command, language="bash")
        st.caption("Direct training entry point.")

        st.subheader("Root CLI equivalent")
        st.code(root_command, language="bash")
        st.caption("Convenient when you prefer single top-level entrypoint commands.")
