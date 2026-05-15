## ML Methods Implemented in ogbl-collab Link Prediction Project

### Overview

This project implements **three link prediction methods** for the **ogbl-collab** dataset. All methods follow the same entry-point convention: receive OGB-style edge splits and node data, produce positive/negative edge scores, and return a standard result dictionary with Hits@K metrics computed via the official OGB Evaluator.

The three methods are registered in **`/home/husam/Desktop/YTU-YL/webmining/ogl-collab-implementation/src/experiments/configs.py`, line 9**:
```python
SUPPORTED_METHODS = ("common_neighbors", "mlp", "gcn")
```

---

### Method 1: Common Neighbors (Baseline)

**File**: `src/methods/common_neighbors.py`

A non-parametric, feature-free baseline. Scores a candidate edge `(u, v)` by counting shared neighbors in the training graph. No node features, no learned parameters.

| Component | Lines | Description |
|-----------|-------|-------------|
| `CommonNeighborsModel` dataclass | 21-26 | Holds fitted state: `adjacency: dict[int, set[int]]` |
| `fit_common_neighbors()` | 62-74 | Builds adjacency map from training edges |
| `_common_neighbor_count()` | 77-84 | Returns `|N(u) intersect N(v)|`, iterates over the smaller neighbor set for efficiency |
| `_tie_breaker()` | 87-89 | Deterministic hash for tie-breaking |
| `score_edges_common_neighbors()` | 92-107 | Scores candidate edges, optionally adds tie-breaker |
| `run_common_neighbors()` | 114-179 | Top-level entry point: fits model, scores valid/test splits, returns metrics |

Configuration (in `configs.py`, lines 44-48): `make_undirected` (default True), `add_tie_breaker` (default True).

---

### Method 2: Feature-Only MLP

**File**: `src/methods/mlp.py`

A neural baseline that uses **only node features** -- no graph structure. Endpoint feature vectors are combined by element-wise multiplication before feeding through an MLP with sigmoid output.

| Component | Lines | Description |
|-----------|-------|-------------|
| `MlpLinkPredictor` class | 21-60 | Stacked Linear+ReLU+Dropout, sigmoid output. Uses `x_i * x_j` (line 53) |
| `train_mlp_epoch()` | 90-130 | One epoch: random negative sampling, BCE loss |
| `run_mlp()` | 158-254 | Top-level entry point: trains N epochs, scores valid/test |

Configuration (in `configs.py`, lines 50-58): `hidden_channels=256`, `num_layers=3`, `dropout=0.0`, `learning_rate=0.01`, `batch_size=65536`, `epochs=200`.

---

### Method 3: GCN (Graph Convolutional Network)

**File**: `src/methods/gcn.py`

A graph neural network using **both node features and graph structure**. GCN encoder produces node embeddings via message passing; MLP decoder scores pairs.

| Component | Lines | Description |
|-----------|-------|-------------|
| `GcnEncoder` class | 16-55 | Stacked GCNConv layers with ReLU+Dropout |
| `GcnLinkPredictor` class | 58-97 | MLP decoder, same arch as MLP. Uses `h_i * h_j` (line 90) |
| `train_gcn_epoch()` | 115-161 | Encodes all nodes, batches train edges, random negatives, gradient clipping |
| `run_gcn()` | 193-303 | Top-level entry point |

Configuration (in `configs.py`, lines 60-68): `hidden_channels=256`, `num_layers=3`, `dropout=0.2`, `learning_rate=0.005`, `batch_size=65536`, `epochs=200`.

---

### Method Dispatch

**`src/experiments/runner.py`**: `_run_method()` (lines 16-54) dispatches by `config.method_name` to the correct method. `run_experiment()` (lines 57-95) loads data, scales splits, runs method, saves result.

### Configuration System

**`src/experiments/configs.py`**: `ExperimentConfig` dataclass (lines 13-30), `get_method_config()` (lines 42-70), `get_default_config()` (lines 73-86).

### Evaluation

**`src/evaluation/metrics.py`**: Uses official OGB Evaluator. Default K values: (10, 50, 100) (line 8).

### Complete File Index

| File | Purpose |
|------|---------|
| `src/methods/common_neighbors.py` | Common Neighbors baseline |
| `src/methods/mlp.py` | Feature-only MLP |
| `src/methods/gcn.py` | GCN with MLP decoder |
| `src/experiments/configs.py` | Default hyperparameters and config |
| `src/experiments/runner.py` | Experiment orchestration |
| `src/experiments/tuning.py` | Grid-search tuning |
| `src/evaluation/metrics.py` | Hits@K evaluation |
| `src/train.py` | CLI entry point |
| `src/data/loader.py` | OGB dataset loading |
