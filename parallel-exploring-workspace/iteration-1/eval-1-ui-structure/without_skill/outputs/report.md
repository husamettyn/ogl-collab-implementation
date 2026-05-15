# UI Structure Report: ogbl-collab Link Prediction Streamlit App

## 1. Entry Points

**`src/dash.py`** (lines 1-15): Thin launcher -- adds project root to `sys.path`, calls `src.ui.app.main()`.

**`src/ui/app.py`** (lines 1-387): Main Streamlit app. `main()` at line 55 sets `layout="wide"`, `initial_sidebar_state="collapsed"`, title `"ogbl-collab | Sunum"`. Custom CSS via `_inject_css()` (lines 13-52). Six top-level tabs at lines 68-75.

---

## 2. Tab Structure

| Tab | Renderer | Location |
|-----|----------|----------|
| "Proje Ozeti" | `_render_overview()` | app.py:96-183 (inline) |
| "Veri Seti Kesfi" | `render_dataset_explorer()` | page_dataset_explorer.py:217 |
| "Metodoloji" | `_render_methodology()` | app.py:185-326 (inline, 4 sub-tabs) |
| "Deney Sonuclari" | `render_results_dashboard()` | page_results_dashboard.py:13 |
| "Sonuc & Tartisma" | `_render_conclusions()` | app.py:329-371 (inline) |
| "Interaktif Demo" | `_render_interactive_demo()` | app.py:374-383 (2 sub-tabs) |

Sub-tabs in Tab 2 (line 189): CN, MLP, GCN, Karsilastirma.
Sub-tabs in Tab 6 (line 377): Graph Explorer (`page_graph_explorer.py:309`), Algorithm Runner (`page_algorithm_runner.py:54`).

---

## 3. Page Details

### Dataset Explorer (`page_dataset_explorer.py:217-317`)
Hero metrics (5 columns, line 236), split pie chart (line 249), weight distribution bar (line 254), year trend line (line 260), degree distribution log-log scatter (line 268), top-20 authors bar (line 273), feature magnitude histogram (line 279), summary table (line 286), description (line 303).

### Results Dashboard (`page_results_dashboard.py:13-111`)
Loads `results/raw/*.json` (line 18), 5 multiselect filters (lines 30-37), KPI cards (line 73), summary table (line 81), best rows table (line 85), loss curve with matplotlib (lines 88-107), generated PNGs (lines 109-111).

### Graph Explorer (`page_graph_explorer.py:309-406`)
Controls (lines 316-337), reservoir sampling up to 6000 edges (line 23), custom force-directed layout (line 109), pydeck WebGL rendering (line 260), degree-band filtering (line 95).

### Algorithm Runner (`page_algorithm_runner.py:54-199`)
Form at line 60: method/scale/preset/seed/device, conditional hyperparams, calls `run_experiment()` at line 168 synchronously.

---

## 4. Shared Utilities

**`src/ui/common.py`** (lines 1-47): `RUNNER_PRESETS` (line 11), LRU-cached `load_bundle()` (line 24), `safe_len()` (line 29), `split_count_rows()` (line 35).

**`src/ui/pages.py`** (lines 1-15): Re-exports 5 renderers. `render_dataset_manager` exported but **unused** by app.py.

**`src/ui/page_dataset_manager.py`** (lines 8-34): Orphaned dataset overview.

---

## Key Findings

1. No sidebar/multipage -- horizontal `st.tabs` only, sidebar collapsed.
2. `page_dataset_manager.py` is dead code (exported but never imported by app.py).
3. Three plotting libs: Plotly, matplotlib, pydeck.
4. Algorithm Runner runs synchronously in Streamlit event loop -- no background execution.
