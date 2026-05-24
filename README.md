# ogbl-collab Link Prediction

This project compares **Common Neighbors**, **MLP**, and **GCN** methods on the
`ogbl-collab` link prediction benchmark (235,868 nodes, 128-dim features,
1,285,465 edges spanning 2005–2020).

The implementation is independent from the local `ogb/` GitHub source tree.
Runtime code uses the installed `ogb` package APIs.

---

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

---

## Run Experiments

### Single method / scale

```bash
# Common Neighbors (heuristic — no training required)
python main.py train --method common_neighbors --scale 0.1

# MLP (feature-only baseline)
python main.py train --method mlp --scale 0.5 --epochs 200

# GCN (graph neural network)
python main.py train --method gcn --scale 1.0 --epochs 200
```

### Full benchmark grid

Run all methods across all scales with one command:

```bash
python main.py benchmark --methods common_neighbors mlp gcn --scales 0.1 0.5 1.0 --seed 42
```

### Multi-seed GCN evaluation

Run GCN with 3 different random seeds for robustness analysis:

```bash
python scripts/run_multi_seed_gcn.py --scales 0.1 0.5 1.0 --seeds 42 123 456
```

### Hyperparameter tuning

```bash
python main.py tune --method gcn --scale 1.0 --max-runs 20
```

---

## Visualization Pipeline

The project offers **two** visualisation tracks: interactive Plotly charts for
the Streamlit dashboard, and publication-quality Matplotlib PDF figures for the
LaTeX report.

```
                      ┌──────────────────────────┐
                      │  results/raw/*.json       │
                      │  (per-experiment results) │
                      └──────────┬───────────────┘
                                 │
                                 ▼
                      ┌──────────────────────────┐
                      │  python main.py assets    │
                      │  ─────────────────────── │
                      │  1. load_results()        │
                      │  2. write_summary_csv()   │
                      │  3. save_all_plots()      │
                      └──────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ results/raw/        │   │ results/plots/       │
        │ summary.csv         │   │ *.png (7 plot types) │
        └─────────┬───────────┘   └─────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ latex/gen_plots.py  │   │ scripts/             │
        │ (matplotlib, PDF)   │   │ generate_latex_tables │
        └─────────┬───────────┘   └──────────┬──────────┘
                  │                          │
                  ▼                          ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ latex/figures/      │   │ stdout (copy into   │
        │ fig*.pdf (4 figs)   │   │ report.tex tables)  │
        └─────────────────────┘   └─────────────────────┘
```

### Generate all report assets

```bash
# 1. Interactive Plotly plots (7 types → results/plots/*.png)
python main.py assets

# 2. LaTeX figure PDFs (4 figures → latex/figures/fig*.pdf)
python latex/gen_plots.py

# 3. LaTeX table snippets (copy into report.tex)
python scripts/generate_latex_tables.py --table all
```

Output includes:

| Step | Output | Location |
|------|--------|----------|
| `main.py assets` | `summary.csv` (flattened results) | `results/raw/summary.csv` |
| `main.py assets` | 7 Plotly PNGs (hits, heatmap, runtime, memory, scatter, training curves, multi-seed) | `results/plots/` |
| `latex/gen_plots.py` | 4 PDF figures (dataset overview, Hits@50 bars, scale analysis, multi-seed) | `latex/figures/` |
| `scripts/generate_latex_tables.py` | 3 LaTeX `tabular` environments (main results, multi-seed, runtime) | stdout |

### Plot reference

See [`results/plots/README.md`](results/plots/README.md) for a complete mapping
of each plot file to its generating function.

---

## Dashboard

Launch the Streamlit dashboard with:

```bash
streamlit run src/dash.py
```

The dashboard includes:

- **Graph Explorer** — graph-level dataset visuals (sampled subgraph, degree
  distribution, split composition, feature analysis)
- **Algorithm Runner** — preset and method-aware parameter controls
- **Results Dashboard** — filtering, KPI cards, best-per-method summaries,
  interactive Plotly charts, and optional loss-curve inspection

---

## LaTeX Report

The project includes a full academic report in [`latex/report.tex`](latex/report.tex).

**To regenerate the report:**
```bash
cd latex
pdflatex report.tex    # run 2–3 times for cross-references
```

**To regenerate all figures and tables from actual data:**
```bash
python latex/gen_plots.py                          # PDF figures
python scripts/generate_latex_tables.py --table all # LaTeX tables (copy into report.tex)
```

---

## Adding a New Plot

1. Add the plotting function in [`src/vis/plots.py`](src/vis/plots.py) (Plotly) or
   [`latex/gen_plots.py`](latex/gen_plots.py) (Matplotlib PDF).
2. Register it in [`save_all_plots()`](src/vis/plots.py:414) for the Plotly track.
3. Add its entry to the plot reference in [`results/plots/README.md`](results/plots/README.md).
4. Reference it in [`latex/report.tex`](latex/report.tex) with `\includegraphics`.

---

## Project Layout

```
.
├── main.py                  # Top-level CLI (train / benchmark / assets / tune)
├── requirements.txt         # Python dependencies
├── README.md                # ← this file
├── ARC.md                   # Architecture guide
├── PLAN.md                  # Implementation plan
├── plans/                   # Task-specific plans & proposals
│   └── visualization-report-plan.md
│
├── src/
│   ├── train.py             # Standalone single-run entry point
│   ├── dash.py              # Streamlit dashboard entry point
│   ├── data/                # Dataset loading & preprocessing
│   ├── methods/             # Common Neighbors, MLP, GCN implementations
│   ├── evaluation/          # Metrics (Hits@K) & runtime/memory measurement
│   ├── experiments/         # Runner, configs, results persistence, tuning
│   ├── ui/                  # Streamlit pages (explorer, runner, dashboard)
│   └── vis/                 # Plotly plotting functions & table helpers
│
├── scripts/
│   ├── run_multi_seed_gcn.py      # Multi-seed GCN benchmark runner
│   └── generate_latex_tables.py   # LaTeX table export from summary.csv
│
├── latex/
│   ├── report.tex           # Academic LaTeX report
│   ├── gen_plots.py         # Data-driven Matplotlib PDF figure generator
│   └── figures/             # Generated PDF figures (fig1–fig4)
│
└── results/
    ├── raw/                 # Per-experiment JSON results + summary.csv
    │   ├── multi_seed/      # Multi-seed GCN run directories
    │   └── tuning/          # Hyperparameter tuning results
    └── plots/               # Generated Plotly PNG plots
        └── README.md        # Plot reference (file → function mapping)
```

Generated outputs belong under `results/`; source modules should not write
generated artifacts into `src/`.
