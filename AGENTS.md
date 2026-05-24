# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## CLI Commands

```bash
# Single experiment
python main.py train --method common_neighbors|mlp|gcn --scale 0.1 --epochs 200

# Full benchmark grid (methods × scales)
python main.py benchmark --methods common_neighbors mlp gcn --scales 0.1 0.5 1.0

# Multi-seed GCN (scripts add project root to sys.path, not main.py)
python scripts/run_multi_seed_gcn.py --seeds 42 123 456

# Hyperparameter tuning (two presets: quick|full)
python main.py tune --method gcn --scale 1.0 --preset quick

# Generate summary CSV + plots from saved results
python main.py assets

# Streamlit dashboard (Turkish UI)
streamlit run src/dash.py
```

**No test framework or test files exist** in this project.

## Non-Obvious Code Patterns

- **PyTorch 2.6+ OGB compatibility**: [`_configure_torch_dataset_loading()`](src/data/loader.py:23) in [`src/data/loader.py`](src/data/loader.py): sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and registers PyG storage classes via `torch.serialization.add_safe_globals()`. Required before any OGB dataset loading on modern PyTorch.

- **Lazy `import torch` pattern**: [`_require_torch()`](src/methods/mlp.py:15) in [`src/methods/mlp.py`](src/methods/mlp.py) does `import torch` inside a function body (not at module level). [`src/methods/gcn.py`](src/methods/gcn.py) imports this from MLP rather than defining its own.

- **`ExperimentConfig`** in [`src/experiments/configs.py`](src/experiments/configs.py:13): uses `@dataclass(slots=True)` with custom `to_dict()` that converts `ks` tuple → list for JSON serialization.

- **Common Neighbors tie-breaker** in [`src/methods/common_neighbors.py`](src/methods/common_neighbors.py:87): deterministic hash `(source * 1_000_003 + target * 97) % 1_000_000 / 1e15` — subtle custom scoring adjustment.

- **Edge tensor shape handling** in [`src/methods/mlp.py`](src/methods/mlp.py:70): [`_edge_tensor()`](src/methods/mlp.py:70) auto-transposes `[2, num_edges]` → `[num_edges, 2]` silently.

- **Memory tracking** in [`src/evaluation/runtime.py`](src/evaluation/runtime.py:29): prefers `psutil` but falls back to `resource.getrusage()` without warning.

- **Streamlit UI is entirely in Turkish** ([`src/ui/app.py`](src/ui/app.py): tabs say "Proje Ozeti", "Veri Seti Kesfi", etc.)

- **Results filename format**: `{timestamp}_{method}_scale_{scale}_seed_{seed}.json` — see [`result_filename()`](src/experiments/results.py:31) in [`src/experiments/results.py`](src/experiments/results.py).

- **Tuning presets** in [`src/experiments/tuning.py`](src/experiments/tuning.py:30): `quick` (small grid) vs `full` (comprehensive grid), controlled via `--preset`.

- **Scripts add sys.path manually** (e.g., [`scripts/run_multi_seed_gcn.py`](scripts/run_multi_seed_gcn.py:19-21)), not importable from `main.py`.

## Critical Defaults (MLP vs GCN)

| Param | MLP | GCN |
|-------|-----|-----|
| `hidden_channels` | 256 | 256 |
| `num_layers` | 3 | 3 |
| `dropout` | 0.0 | 0.2 |
| `learning_rate` | 0.01 | 0.005 |
| `batch_size` | 65536 | 65536 |
| `epochs` | 200 | 400 |
