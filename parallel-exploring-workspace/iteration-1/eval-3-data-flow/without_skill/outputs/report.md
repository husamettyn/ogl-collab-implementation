# Data Flow Report: ogbl-collab Link Prediction Project

## 1. Entry Points

Two entry points converge on `run_experiment` (`src/experiments/runner.py:57`):
- `main.py` (lines 91-177): CLI with train/benchmark/tune/assets subcommands
- `src/train.py` (lines 78-97): Standalone CLI for single experiments

Both parse into `ExperimentConfig` dataclass (`configs.py:13-30`).

---

## 2. Dataset Loading (Stage 1)

**`load_collab_data_bundle()`** at `src/data/loader.py:104-119`:
1. `load_collab_dataset()` (lines 86-95): `PygLinkPropPredDataset(name="ogbl-collab")` with PyTorch 2.6+ compatibility (lines 23-83).
2. `dataset[0]` (line 111): Extracts `data.x` [235868x128], `data.edge_index` [2x1179052], edge_weight, edge_year.
3. `get_edge_split()` (lines 98-101): Time-based train/valid/test splits.

**`make_scaled_split()`** at `src/data/preprocessing.py:101-119`: Scales training edges by `sample_count = max(1, int(edge_count * scale))` using deterministic randperm(seed=42). Scales: 0.1, 0.5, 1.0.

---

## 3. Training (Stage 2)

### Common Neighbors (`src/methods/common_neighbors.py`)
Receives only `split_edge`. Builds adjacency `dict[int, set[int]]` from train edges (lines 62-74). No training loop.

### MLP (`src/methods/mlp.py`)
Receives `data` and `split_edge`. `MlpLinkPredictor` (lines 21-60): element-wise multiplication of endpoint features, then stacked Linear+ReLU+Dropout, sigmoid output. Per-epoch: random negative sampling BCE loss (lines 90-130).

### GCN (`src/methods/gcn.py`)
Receives `data`, `split_edge`, plus message-passing edge index from `_make_message_passing_edge_index()` (lines 107-112): bidirectional train edges `[2, 2*E]`.
- `GcnEncoder` (lines 16-55): Stacked GCNConv layers
- `GcnLinkPredictor` (lines 58-97): MLP on encoder embeddings, same architecture as MLP predictor
- Full-graph message passing each epoch, batch edge scoring, gradient clipping at 1.0 (lines 115-161)

---

## 4. Evaluation (Stage 3)

**`evaluate_link_prediction()`** at `src/evaluation/metrics.py:62-86`:
1. For valid/test splits, scores positive and negative edges
2. `compute_hits_at_k()` (lines 28-45): Uses official `ogb.linkproppred.Evaluator(name="ogbl-collab")` for K=(10,50,100)

**`track_resources()`** at `src/evaluation/runtime.py:53-62`: psutil RSS memory + wall clock timing.

---

## 5. Result Persistence (Stage 4)

**`save_result()`** at `src/experiments/results.py:40-57`: Serializes result dict to `results/raw/{timestamp}_{method}_scale_{scale}_seed_{seed}.json`.

**`aggregate_results()`** at `results.py:75-98`: Flattens into one row per split. Written to `results/raw/summary.csv`.

**`save_all_plots()`** at `src/vis/plots.py:105-112`: Hits@50, runtime, memory comparison PNGs.

---

## Complete Data Flow

```
CLI/UI -> ExperimentConfig -> run_experiment()
  -> load_collab_data_bundle() -> PygLinkPropPredDataset -> data.x, edge_index, split_edge
  -> make_scaled_split() -> scaled train edges
  -> _run_method() -> run_{cn,mlp,gcn}()
       -> training (adjacency for CN, MLP epochs, GCN epochs)
       -> scoring (valid/test positive + negative scores)
       -> evaluate_link_prediction() -> Evaluator.eval() -> Hits@K
  -> Result dict -> save_result() -> results/raw/*.json
                 -> aggregate_results() -> summary.csv
                 -> save_all_plots() -> plots/*.png
```

## Key Design Properties

1. Scaling is deterministic and applied before any method sees data.
2. GCN message-passing edges come only from training split -- no test leakage.
3. Training negatives are random (fresh each batch); evaluation negatives are pre-computed from OGB splits.
4. Official OGB Evaluator used for all metric computation.
