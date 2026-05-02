# ogbl-collab Link Prediction

This project compares Common Neighbors, MLP, and GCN methods on the
`ogbl-collab` link prediction benchmark.

The implementation is independent from the local `ogb/` GitHub source tree.
Runtime code uses the installed `ogb` package APIs.

## Setup

Create and install the environment with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

You can also run commands without activating:

```bash
uv run python src/train.py --help
```

## Run Experiments

Common Neighbors:

```bash
python src/train.py --method common_neighbors --scale 0.1
```

MLP smoke run:

```bash
python src/train.py --method mlp --scale 0.1 --epochs 1 --batch-size 4096
```

GCN smoke run:

```bash
python src/train.py --method gcn --scale 0.1 --epochs 1 --batch-size 4096
```

Root CLI equivalent:

```bash
python main.py train --method common_neighbors --scale 0.1
```

## Generate Report Assets

After saving experiment results under `results/raw/`, generate summary and plots:

```bash
python main.py assets
```

This creates:

- `results/raw/summary.csv`
- `results/plots/hits_at_50_comparison.png`
- `results/plots/runtime_comparison.png`
- `results/plots/memory_comparison.png`

## Dashboard

Launch the Streamlit dashboard with:

```bash
streamlit run src/dash.py
```

The dashboard includes:

- Dataset Manager
- Algorithm Runner
- Results Dashboard

## Project Layout

See `ARC.md` for the architecture guide and `PLAN.md` for the implementation
plan.

Generated outputs belong under `results/`; source modules should not write
generated artifacts into `src/`.