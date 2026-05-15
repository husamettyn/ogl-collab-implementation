# UI Structure: Exploration Results

## Summary

The project's UI is a single-page Streamlit application with six top-level tabs and nested sub-tabs that serve as a presentation layer for the ogbl-collab link prediction research project. Navigation is entirely tab-based (no sidebar, no multi-page routing), with all page rendering logic spread across six dedicated modules under `src/ui/`.

---

## 1. App Entry Point and Navigation

**File:** `src/dash.py`, lines 7-15 -- Launches via `streamlit run src/dash.py`, calls `src/ui/app.py:main()`.

**File:** `src/ui/app.py`, lines 55-93 -- `st.set_page_config` with `layout="wide"`, `initial_sidebar_state="collapsed"`, title `"ogbl-collab | Sunum"`.

Six top-level tabs at line 68:
| Tab | Label | Renderer | Module |
|-----|-------|----------|--------|
| 0 | "Proje Ozeti" | `_render_overview()` | inline app.py |
| 1 | "Veri Seti Kesfi" | `render_dataset_explorer()` | page_dataset_explorer.py |
| 2 | "Metodoloji" | `_render_methodology()` | inline, 4 sub-tabs |
| 3 | "Deney Sonuclari" | `render_results_dashboard()` | page_results_dashboard.py |
| 4 | "Sonuc & Tartisma" | `_render_conclusions()` | inline |
| 5 | "Interaktif Demo" | `_render_interactive_demo()` | inline, 2 sub-tabs |

Sub-tabs: Tab 2 at line 189 (CN, MLP, GCN, Karsilastirma). Tab 5 at line 377 (Graph Explorer from `page_graph_explorer.py:309`, Algorithm Runner from `page_algorithm_runner.py:54`).

CSS: `_inject_css()` at lines 13-52 -- hides Streamlit header, adjusts padding, styles tabs with blue highlight `#4e79a7`.

---

## 2. Individual Pages

### Tab 0 -- "Proje Ozeti" (lines 96-182)
Static presentation: problem description, dataset stats (235,868 researchers, 1,179,052 collaborations, 128-dim embeddings, 1963-2017), methods table, 3-column architecture, CLI examples.

### Tab 1 -- "Veri Seti Kesfi" (`page_dataset_explorer.py:217-317`)
Seven Plotly visualizations: hero metrics (line 236), split pie chart (line 249), weight distribution (line 254), year trend (line 260), degree distribution log-log (line 268), top-20 authors (line 273), feature magnitude histogram (line 279). Summary table (line 286-299).

### Tab 2 -- "Metodoloji" (lines 185-326)
Four inline sub-tabs covering CN heuristic (lines 209-237), MLP architecture (lines 240-269), GCN message passing (lines 273-299), comparison table (lines 301-326).

### Tab 3 -- "Deney Sonuclari" (`page_results_dashboard.py:13-111`)
Loads results from `results/raw/` (line 18), 5 multiselect filters (lines 30-37), KPI metric cards (lines 73-78), summary table (line 81), best rows table (line 85), loss curve with matplotlib (lines 88-107), generated plots display (lines 109-111).

### Tab 4 -- "Sonuc & Tartisma" (lines 329-371)
Static: 4 findings, trade-off table, future work, summary.

### Tab 5 -- "Interaktif Demo"
- Graph Explorer (`page_graph_explorer.py:309-406`): pydeck WebGL graph, reservoir sampling up to 6000 edges (line 23), custom force-directed layout (line 109), degree-band filtering (line 95).
- Algorithm Runner (`page_algorithm_runner.py:54-199`): form with method/scale/preset/seed/device, conditional hyperparameters, `run_experiment()` execution (line 168), CLI command display (line 192).

---

## 3. Shared Utilities

**`src/ui/common.py`** (lines 1-47): `RUNNER_PRESETS` (line 11), `load_bundle()` with LRU cache (line 24), `safe_len()` (line 29), `split_count_rows()` (line 35).

**`src/ui/pages.py`** (lines 1-15): Re-exports 5 renderers. `render_dataset_manager` is exported but **never called** by `app.py`.

**`src/ui/page_dataset_manager.py`** (lines 8-34): Orphaned -- simpler dataset overview page not wired into the app.

---

## Key Findings

1. No sidebar/multipage routing -- entirely horizontal `st.tabs` with collapsed sidebar (line 63).
2. `page_dataset_manager.py` is orphaned -- exported from pages.py but never rendered.
3. Three plotting libraries: Plotly, matplotlib, pydeck/deck.gl.
4. Only Algorithm Runner (Tab 5b) triggers computation; all other tabs are static.
5. Turkish tab labels for presentation-oriented tabs, English for interactive components.
