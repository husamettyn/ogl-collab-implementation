# Architect Mode — Project Architecture Rules

Non-obvious architectural constraints discovered by analyzing the codebase.

## Data Flow Architecture

```
main.py / src/train.py
        │
        ▼
src/experiments/runner.py::run_experiment(config)
        │
        ├── 1. src/data/loader.py::load_collab_data_bundle()
        │         └── calls _configure_torch_dataset_loading() FIRST (critical)
        │
        ├── 2. src/data/preprocessing.py::make_scaled_split()
        │         └── deterministic torch.randperm with seeded generator
        │
        ├── 3. _run_method() dispatches to:
        │         ├── src/methods/common_neighbors.py (no training needed)
        │         ├── src/methods/mlp.py (feature-only, no graph structure)
        │         └── src/methods/gcn.py (graph neural network)
        │
        └── 4. src/experiments/results.py::save_result() (if save_result=True)
                └── JSON file at results/raw/{timestamp}_{method}_scale_{scale}_seed_{seed}.json
```

## Key Architectural Constraints

1. **Dataset loading is centralized in `src/data/`**: No module outside `src/data/` should import OGB or PyG dataset classes directly. The project-level interface is [`CollabDataBundle`](../../src/data/loader.py:13).

2. **Methods are isolated and stateless**: Each method file exposes a `run_*()` function that takes data + config and returns a result dict. No shared state between runs.

3. **No cross-method inheritance**: MLP and GCN share `_require_torch()` and `_edge_tensor()` only because GCN imports them from the MLP module. This is a code-sharing shortcut, not an architectural hierarchy.

4. **Results are append-only**: Saved JSON files are never overwritten. Each run produces a unique filename via timestamp. Deduplication happens later in analysis (e.g., [`_deduplicate_by_seed()`](../../scripts/generate_latex_tables.py:65)).

5. **Scripts are NOT part of the main module tree**: They add `sys.path` at runtime. If you need script functionality from `main.py`, refactor into `src/`.

## Method Comparison Architecture

All three methods produce the same output schema:
```python
{
    "method_name": str,
    "dataset_scale": float,
    "seed": int,
    "runtime_seconds": float,
    "memory_mb": float,
    "memory_delta_mb": float,
    "metrics": {
        "valid": {"hits_at_10": float, "hits_at_50": float, "hits_at_100": float},
        "test": {"hits_at_10": float, "hits_at_50": float, "hits_at_100": float},
    },
    "val_metrics": [{"epoch": int, "hits_at_50": float}, ...],  # MLP/GCN only
    "experiment_config": dict,
}
```

## Streamlit Dashboard Architecture

```
src/dash.py  →  src/ui/app.py::main()
                     │
                     ├── tab1: Proje Ozeti (overview)
                     ├── tab2: Veri Seti Kesfi (dataset explorer)
                     ├── tab3: Metodoloji (methodology)
                     ├── tab4: Deney Sonuclari (results dashboard)
                     ├── tab5: Sonuc & Tartisma (conclusion)
                     └── tab6: Interaktif Demo (interactive runner)
```

All tabs are in Turkish. The UI reads from `results/raw/*.json` files directly.

## Tuning Architecture

```
main.py tune → src/experiments/tuning.py::tune_method()
                   │
                   ├── Generates configs from Cartesian product of hyperparameter grid
                   ├── Grid controlled by preset: "quick" (2-4 values) or "full" (3-4 values)
                   ├── Each config → run_experiment() → saved to results/raw/tuning/{id}/
                   └── Best config selected by max Hits@50 on valid split
```

## Critical Dependency Graph

- `src.experiments.runner` → `src.data.loader`, `src.data.preprocessing`, `src.methods.*`
- `src.methods.gcn` → `src.methods.mlp` (imports `_require_torch`, `_edge_tensor`)
- `src.evaluation.metrics` → `ogb.linkproppred.Evaluator` (lazy import)
- `src.vis.plots` → `src.vis.tables`
- No circular dependencies present.
