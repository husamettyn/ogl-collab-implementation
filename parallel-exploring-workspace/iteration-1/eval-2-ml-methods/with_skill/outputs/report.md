# ML Methods Implemented: Exploration Results

## Summary

This repository implements **three link prediction methods** for the OGB ogbl-collab dataset: a non-learned Common Neighbors heuristic, a feature-only MLP, and a GCN-based graph neural network. All share a common evaluation harness (Hits@K via OGB Evaluator), runtime tracking, and a CLI/experiment runner.

---

## 1. Common Neighbors (Heuristic Baseline)

**File:** `src/methods/common_neighbors.py`

Scores candidate edge (u,v) by counting shared neighbors in training graph. No features, no learned parameters.

| Component | Lines | Purpose |
|-----------|-------|---------|
| `CommonNeighborsModel` dataclass | 22-25 | Holds adjacency map |
| `fit_common_neighbors()` | 62-74 | Builds `dict[int, set[int]]` |
| `_common_neighbor_count()` | 77-84 | Set intersection |
| `score_edges_common_neighbors()` | 92-107 | Scores edges + tie-breaker |
| `run_common_neighbors()` | 114-179 | Top-level entry point |

Config (`configs.py:44-48`): `make_undirected=True`, `add_tie_breaker=True`.

---

## 2. MLP (Feature-Only Neural Baseline)

**File:** `src/methods/mlp.py`

Uses only node features, no graph structure. Element-wise multiplication of endpoint features (`x_i * x_j`, line 53).

| Component | Lines | Purpose |
|-----------|-------|---------|
| `MlpLinkPredictor` class | 21-60 | Stacked Linear+ReLU+Dropout, sigmoid output |
| `train_mlp_epoch()` | 90-130 | Random negative sampling, BCE loss |
| `run_mlp()` | 158-254 | Top-level entry point |

Default hyperparams (`configs.py:50-58`): hidden_channels=256, num_layers=3, dropout=0.0, lr=0.01, batch_size=65536, epochs=200.

---

## 3. GCN (Graph Neural Network)

**File:** `src/methods/gcn.py`

Two-module architecture: GCN encoder (message passing) + MLP predictor. Only method using graph structure.

| Component | Lines | Purpose |
|-----------|-------|---------|
| `GcnEncoder` class | 16-55 | Stacked GCNConv + ReLU + Dropout |
| `GcnLinkPredictor` class | 58-97 | MLP on GCN embeddings, uses `h_i * h_j` (line 90) |
| `train_gcn_epoch()` | 115-161 | Full-graph encode, batch negative sampling, gradient clipping |
| `run_gcn()` | 193-303 | Top-level entry point |

Default hyperparams (`configs.py:60-68`): hidden_channels=256, num_layers=3, dropout=0.2, lr=0.005, batch_size=65536, epochs=200, grad clip=1.0.

---

## 4. Shared Infrastructure

- **Config**: `ExperimentConfig` dataclass (`configs.py:13-30`), `SUPPORTED_METHODS` at line 9.
- **Dispatch**: `_run_method()` in `runner.py:16-54` -- if/elif chain per method.
- **Evaluation**: `evaluate_link_prediction()` in `metrics.py:62` -- OGB Evaluator, K=(10,50,100).
- **Tuning**: Grid search for MLP/GCN in `tuning.py:21-257`. CN not tunable.
- **Runtime**: `track_resources()` in `runtime.py:53-62` -- psutil RSS + wall clock.

## Key Observations

1. CN requires no `data.x`; MLP and GCN raise ValueError if features absent.
2. Only GCN uses graph structure (message passing).
3. Both neural methods use element-wise multiplication for edge combination.
4. Both neural methods use identical BCE loss: `-log(pos) - log(1-neg)`.
5. GCN uses full-graph message passing each epoch -- no mini-batch sampling of neighbors.
