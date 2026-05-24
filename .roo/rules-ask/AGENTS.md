# Ask Mode — Project Documentation & Context

Non-obvious documentation context discovered by reading the codebase.

## Counterintuitive Code Organization

- **Streamlit UI is entirely in Turkish** ([`src/ui/app.py`](../../src/ui/app.py:83-90)) — tab labels, headings, and UI text all use Turkish ("Proje Ozeti", "Veri Seti Kesfi", "Metodoloji", "Deney Sonuclari", "Sonuc & Tartisma", "Interaktif Demo"). Dashboard code has no English locale option.
- **`src/dash.py` is a thin compatibility launcher** — imports and calls `main()` from [`src/ui/app.py`](../../src/ui/app.py). The real dashboard lives in `src/ui/`.
- **`scripts/` are standalone executables** — they add `sys.path` manually and are NOT importable from `main.py`. Running them requires being at project root.
- **`src/train.py` duplicates** CLI parsing from [`main.py`](../../main.py) — both are entry points with similar but slightly different argparse definitions.

## Hidden Dependencies & Coupling

- **GCN imports from MLP**: [`GcnEncoder`](../../src/methods/gcn.py:16) uses `_require_torch` and `_edge_tensor` imported from [`src.methods.mlp`](../../src/methods/mlp.py:10) — not from a shared utility module.
- **Common Neighbors** has its own [`iter_edge_pairs()`](../../src/methods/common_neighbors.py:43) — a duplicate of the one in [`src/data/preprocessing.py`](../../src/data/preprocessing.py:27). Both coexist.
- **`ExperimentConfig.hyperparameters`** dict is passed directly to method-specific functions — schema depends on method name. For `common_neighbors`: `make_undirected` and `add_tie_breaker`. For `mlp`/`gcn`: standard training params.
- **`save_all_plots()`** imports `make_summary_table` from [`src.vis.tables`](../../src/vis/tables.py) (not yet created in file listing — may be generated at runtime).

## Defaults That Differ From Expectation

- **MLP default epochs = 200**, **GCN default epochs = 400** ([`get_method_config()`](../../src/experiments/configs.py:42-68)). GCN trains for twice as many epochs by default.
- **MLP has zero dropout** by default (0.0), GCN has 0.2. GCN's higher regularization compensates for more complex architecture.
- **`batch_size` defaults to `64 * 1024` = 65536** for both MLP and GCN — full-batch-adjacent batch sizing.
- **Dataset scales** are 0.1, 0.5, 1.0 — corresponding to ~10%, 50%, and 100% of training edges.

## Misleading File/Module Names

- **`src/experiments/progress.py`** is NOT about experiment progress tracking — it's a simple `tqdm` wrapper. Multi-seed progress is in `scripts/run_multi_seed_gcn.py`.
- **`src/vis/`** contains Plotly (not matplotlib) code despite the old project history mentioning matplotlib.
- **`src/evaluation/runtime.py`** measures both time AND memory, despite the name suggesting only runtime.
