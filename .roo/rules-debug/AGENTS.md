# Debug Mode — Project Debugging Rules

Non-obvious debugging discoveries from the codebase.

## OGB/PyTorch Loading Issues

- **Most common failure**: `RuntimeError: Weights not found` or pickle errors on PyTorch 2.6+ — check [`_configure_torch_dataset_loading()`](../../src/data/loader.py:23) is being called. This sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and registers PyG storage globals.
- If OGB dataset fails to load with "unsafe globals", the [`safe` list in `_configure_torch_dataset_loading()`](../../src/data/loader.py:56-77) may need additional PyG classes added.
- Dataset root is [`DATASET_DIR`](../../src/experiments/paths.py:11) = `PROJECT_ROOT / "dataset"`. Deletion forces re-download.

## Silent Failures & Gotchas

- **Memory tracking** in [`get_memory_usage_mb()`](../../src/evaluation/runtime.py:29): falls back from `psutil` to `resource.getrusage()` without any warning. On systems without `psutil`, memory values come from `ru_maxrss` (different semantics).
- **Lazy torch import** in [`_require_torch()`](../../src/methods/mlp.py:15): if CUDA is unavailable, [`_select_device()`](../../src/methods/mlp.py:64) silently falls back to CPU — no error raised.
- **Common Neighbors tie-breaker** in [`_tie_breaker()`](../../src/methods/common_neighbors.py:87) adds a tiny deterministic value (max ~1e-9) — can cause confusion if comparing raw scores vs official evaluator.
- **`no_save` flag**: experiments with `--no-save` skip result persistence entirely — results won't appear in `summary.csv` or `main.py assets`.

## Logging & Diagnostics

- Log level is `INFO` by default ([`configure_logging()`](../../src/experiments/runtime_config.py:9)). No debug-level output unless explicitly configured.
- Known warnings suppressed in [`suppress_known_warnings()`](../../src/experiments/runtime_config.py:18) — if something seems wrong, check this file first.
- [`progress_bar()`](../../src/experiments/progress.py:10) wraps `tqdm.auto` — if `tqdm` is missing, iterables run silently (no output, no error).

## Common Errors

- **Scale value out of range**: All experiment commands validate `0 < scale <= 1` (see [`validate_config()`](../../src/experiments/configs.py:35)). Values outside this range raise `ValueError`.
- **Method not found**: Only `common_neighbors`, `mlp`, `gcn` are supported — see [`SUPPORTED_METHODS`](../../src/experiments/configs.py:9).
- **Tuning on unsupported method**: Only `mlp` and `gcn` support tuning — see [`TUNABLE_METHODS`](../../src/experiments/tuning.py:21).
- **Script import errors**: Scripts in [`scripts/`](../../scripts/) add project root to `sys.path` at runtime. Running them from wrong directory breaks imports — always run from project root.
