# Code Mode — Project Coding Rules

Only non-obvious patterns discovered by analyzing the codebase.

## Imports & Module Conventions

- Use `from __future__ import annotations` at top of files for deferred evaluation (used in most modules, e.g., [`src/experiments/benchmark.py`](../../src/experiments/benchmark.py:3))
- [`_require_torch()`](../../src/methods/mlp.py:15) lazy-imports `torch` inside a function — **do not** import torch at module level for method files. Import `_require_torch` from [`src.methods.mlp`](../../src/methods/mlp.py) (GCN does this too).
- [`_edge_tensor()`](../../src/methods/mlp.py:70) and similar helpers tolerate both `[2, num_edges]` and `[num_edges, 2]` shapes — always use `_edge_tensor()` rather than raw `.t()`.
- Lazy imports inside functions (e.g., [`ogb.linkproppred`](../../src/evaluation/metrics.py:35) inside `compute_hits_at_k`) for soft dependencies.
- Use `try/except ImportError` for optional deps (e.g., [`tqdm`](../../src/experiments/progress.py:13), [`psutil`](../../src/evaluation/runtime.py:32)).

## Configuration

- [`ExperimentConfig`](../../src/experiments/configs.py:13) is `@dataclass(slots=True)` with a custom [`to_dict()`](../../src/experiments/configs.py:26) that converts `ks` tuple → list. Use `to_dict()` for JSON serialization, not `asdict()` directly.
- Default hyperparams for MLP vs GCN differ in `dropout`, `learning_rate`, and `epochs` — see [`get_method_config()`](../../src/experiments/configs.py:42). Don't assume they match.

## Data & Preprocessing

- Always call [`_configure_torch_dataset_loading()`](../../src/data/loader.py:23) before loading any OGB dataset on PyTorch 2.6+.
- Dataset scaling uses deterministic [`torch.randperm`](../../src/data/preprocessing.py:90) with a seeded generator — reproducible by design.
- Common Neighbors tie-breaker in [`_tie_breaker()`](../../src/methods/common_neighbors.py:87) uses a deterministic hash formula: `(source * 1_000_003 + target * 97) % 1_000_000 / 1e15`.

## Persistence

- Results filename format: [`result_filename()`](../../src/experiments/results.py:31) → `{timestamp}_{method}_scale_{scale}_seed_{seed}.json`.
- Paths defined in [`src/experiments/paths.py`](../../src/experiments/paths.py) — always import from there, never hardcode paths.
- Scripts in [`scripts/`](../../scripts/) add project root to `sys.path` manually; they are **not** importable via `main.py`.

## Streamlit UI

- UI language is **Turkish** exclusively (tabs: "Proje Ozeti", "Veri Seti Kesfi", etc. in [`src/ui/app.py`](../../src/ui/app.py:83-90)).
- Dashboard launched via `streamlit run src/dash.py`, not `python main.py`.

## Memory & Runtime

- [`track_resources()`](../../src/evaluation/runtime.py:54) is a context manager that captures start/end memory and wall time. Use it to wrap experiment blocks.
- Memory prefers `psutil` but silently falls back to `resource.getrusage()` — no warning emitted.
