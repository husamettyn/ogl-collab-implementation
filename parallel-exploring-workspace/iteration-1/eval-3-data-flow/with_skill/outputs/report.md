# Data Flow: End-to-End Trace

## Summary

The project implements link prediction on the OGB ogbl-collab dataset with three methods. Data flows through four stages: loading, scaling, training/scoring, and evaluation/persistence. Orchestrated by `run_experiment()` at `src/experiments/runner.py:57`.

---

## 1. Dataset Loading

**`load_collab_data_bundle()`** at `src/data/loader.py:104-119`:
1. `load_collab_dataset()` (line 110): `PygLinkPropPredDataset(name="ogbl-collab")` with PyTorch 2.6+ compatibility shims (lines 23-83)
2. `dataset[0]` (line 111): PyG Data with `data.x` [num_nodes, 128], `data.edge_index` [2, num_edges], `data.edge_weight`
3. `load_edge_split()` (line 112): Nested dict `{"train": {"edge": [...], "edge_neg": [...]}, "valid": {...}, "test": {...}}`

Wrapped in `CollabDataBundle` (lines 14-20).

---

## 2. Training Data Scaling

**`make_scaled_split()`** at `src/data/preprocessing.py:101-119`:
- Only scales `"train"` positive edges (line 105)
- `sample_count = max(1, int(edge_count * scale))` (0.1, 0.5, or 1.0)
- Deterministic via `torch.randperm(seed=42)`

---

## 3. Method Routing and Training

**`_run_method()`** at `src/experiments/runner.py:16-54` dispatches by `config.method_name`.

### Common Neighbors (`src/methods/common_neighbors.py:114-179`)
- Fit: builds `dict[int, set[int]]` adjacency from train edges (lines 126-133)
- Score: shared neighbor count per edge pair (lines 138-156)
- No learned parameters

### MLP (`src/methods/mlp.py:158-254`)
- `MlpLinkPredictor` (lines 21-60): `x_i * x_j` -> Linear+ReLU+Dropout -> Sigmoid
- Training (lines 90-130): DataLoader with batch_size=64K, random negative sampling via `torch.randint()`, BCE loss
- Scoring (lines 133-155): eval mode, batched

### GCN (`src/methods/gcn.py:193-303`)
- `GcnEncoder` (lines 16-55): Stacked GCNConv with bidirectional edges (lines 107-112)
- `GcnLinkPredictor` (lines 58-97): MLP on node embeddings, same architecture as MLP
- Training (lines 115-161): Full-graph message passing `h = encoder(x, edge_index)`, batched predictor head, gradient clip 1.0
- Scoring (lines 164-190): encode once, batch score

---

## 4. Evaluation

**`evaluate_link_prediction()`** at `src/evaluation/metrics.py:62-86`:
- `compute_hits_at_k()` (lines 28-45): OGB `Evaluator(name="ogbl-collab")` for k in (10, 50, 100)
- Returns nested metrics dict per split

**`track_resources()`** at `src/evaluation/runtime.py:53-62`: psutil RSS + wall clock

---

## 5. Results Persistence

- **`save_result()`** (`results.py:40-57`): JSON to `results/raw/{timestamp}_{method}_scale_{s}_seed_{s}.json`
- **`aggregate_results()`** (`results.py:75-98`): Flattened rows
- **`write_summary_csv()`** (`benchmark.py:39-59`): `results/raw/summary.csv`
- **`save_all_plots()`** (`vis/plots.py:105-112`): Hits@50, runtime, memory PNGs to `results/plots/`

---

## End-to-End Flow

```
CLI/UI -> ExperimentConfig -> run_experiment()
  -> load_collab_data_bundle() -> PyG Data + split_edge
  -> make_scaled_split() -> scaled train edges
  -> _run_method() -> {cn,mlp,gcn} training + scoring
  -> evaluate_link_prediction() -> Hits@K metrics
  -> result dict -> save_result() -> JSON
                 -> aggregate -> CSV
                 -> plots -> PNG
```

## Key Architectural Observations

1. Only training positive edges are scaled; valid/test are always full-size.
2. GCN uses two edge structures: bidirectional MP edges + directed train edges for loss.
3. Training negatives are random (fresh each batch); evaluation negatives are pre-computed OGB splits.
4. GCN runs full-graph message passing each epoch; DataLoader batching only for the predictor head.
5. Tuning (`tuning.py`) reuses the same `run_experiment()` pipeline in a grid search.
