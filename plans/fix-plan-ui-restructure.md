# UI Restructure Plan: 6 Tabs → 3 Tabs

## Goal

Reduce the Streamlit UI from 6 tabs to 3 focused tabs. Keep only:
1. **Veri Seti Keşfi** — Dataset explorer + Graph explorer (merged)
2. **Algoritma Çalıştırıcı** — Algorithm runner
3. **Sonuçlar** — Results dashboard

Remove unused internal functions from `app.py` but keep the page module files intact.

---

## Changes

### 1. [`src/ui/app.py`](src/ui/app.py) — Restructure to 3 tabs

#### Imports (lines 5-11)
Keep:
- `render_dataset_explorer` — for Tab 1
- `render_algorithm_runner` — for Tab 2
- `render_results_dashboard` — for Tab 3
- `render_graph_explorer` — to embed inside Tab 1

Remove:
- `render_dataset_manager` — already removed

#### main() function (lines ~84-113)
Replace 6 tabs with 3:

| Tab | Label | Content |
|-----|-------|---------|
| tab1 | "Veri Seti Kesfi" | `render_dataset_explorer()` + then `render_graph_explorer()` (either as sub-tabs or sequentially) |
| tab2 | "Algoritma Calistirici" | `render_algorithm_runner()` |
| tab3 | "Sonuclar" | `render_results_dashboard()` |

#### Remove internal functions (keep file, just delete the function definitions from app.py):
- `_render_overview()` — lines 116-202
- `_render_methodology()` — lines 205-322 (includes `_render_cn_method`, `_render_mlp_method`, `_render_gcn_method`, `_render_method_comparison`)
- `_render_conclusions()` — lines 348-391
- `_render_interactive_demo()` — lines 394-403 (but its content — graph_explorer and algorithm_runner — are redistributed)

#### Tab 1: Dataset Explorer + Graph Explorer integration
Option A (recommended — simpler): Show `render_dataset_explorer()` first, then `render_graph_explorer()` below it with a divider like `st.divider()` or `st.markdown("---")`.

Option B: Use `st.tabs(["Veri Seti Kesfi", "Graf Goruntuleyici"])` inside tab1.

**Recommendation: Option A** — simpler and avoids nested tab visual confusion. The graph explorer already has its own internal controls (select boxes, radio buttons, pydeck chart). It works standalone below the dataset explorer content.

### 2. [`README.md`](README.md) — Update dashboard description

Change the "Dashboard (Turkish UI)" section from 6-tab to 3-tab descriptions.

---

## Files NOT modified (intact):
- `src/ui/page_dataset_explorer.py` — kept
- `src/ui/page_algorithm_runner.py` — kept
- `src/ui/page_results_dashboard.py` — kept
- `src/ui/page_graph_explorer.py` — kept (now imported in app.py)
- `src/ui/page_dataset_manager.py` — kept but not imported
- `src/ui/pages.py` — kept (compatibility exports)

---

## Verification

After changes:
```bash
streamlit run src/dash.py
# Should show exactly 3 tabs:
# 1. Veri Seti Kesfi (dataset explorer + graph explorer)
# 2. Algoritma Calistirici (algorithm runner)
# 3. Sonuclar (results dashboard)
```
