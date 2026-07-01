# Fix Plan — BLM5121 Web Mining Term Project

## Overview

This plan details the implementation steps to fix all identified gaps between the current codebase and the instructor's requirements. Each step is self-contained and executable independently.

---

## Step 1: Fix Student ID in Report ✅
**File:** `latex/report.tex` line 28
**Change:** `25501003` → `25501005`

## Step 2: Integrate Dataset Manager into Streamlit UI
**File:** `src/ui/app.py`
**Change:** Replace current "Veri Seti Kesfi" tab (which uses `render_dataset_explorer()`) with `render_dataset_manager()` to satisfy the instructor's exact requirement for "Dataset Manager" screen name. Keep the dataset explorer content inside the manager or add as a sub-section.

Alternatively: Add `render_dataset_manager()` as the primary content of the overview tab and keep explorer as-is.

## Step 3: Add "Run All Three Methods" Feature
**File:** `src/ui/page_algorithm_runner.py`
**Change:** Add a "Run All Methods" button/checkbox that:
1. Takes the current scale, seed, device settings
2. Runs all 3 methods sequentially (Common Neighbors → MLP → GCN)
3. Displays results in a side-by-side comparison table
4. Shows progress for each method

## Step 4: Add System Architecture Diagram to Report
**File:** `latex/report.tex` (after section 4)
**Change:** Add a Mermaid diagram converted to PDF, or use TikZ/pgf to draw the architecture. The diagram should show:
- Data flow: OGB dataset → Data Module → Methods → Evaluation → Results Persistence → GUI/Report
- The diagram can be generated with `latex/gen_plots.py` or added manually

## Step 5: Add GUI Screenshots to Report
**File:** `latex/report.tex`
**Change:** Add 3 figures with screenshots of the Streamlit dashboard:
1. Dataset Manager screen
2. Algorithm Runner screen  
3. Results Dashboard screen

## Step 6: Add Statistical Significance Testing
**Files:** `src/evaluation/` (new file or add to `metrics.py`)
**Change:** Implement statistical significance functions:
- `compute_mann_whitney_test()` for comparing two methods
- `compute_bootstrap_ci()` for confidence intervals
- Update `scripts/generate_latex_tables.py` to include significance test results
- Update report with significance analysis

## Step 7: Add ROC Curve Generation
**Files:** `src/vis/plots.py`
**Change:** Add `plot_roc_curves()` function that:
- For each method at each scale, computes TPR and FPR at various thresholds
- Plots ROC curves for all methods on the same axis
- Includes AUC scores in the legend
- Saves to `results/plots/roc_curves.png`

## Step 8: Fix GCN Fallback Defaults
**File:** `src/methods/gcn.py`
**Changes:**
- Line 228: `dropout=hyperparameters.get("dropout", 0.0)` → `dropout=hyperparameters.get("dropout", 0.2)`
- Line 239: `lr=hyperparameters.get("learning_rate", 0.001)` → `lr=hyperparameters.get("learning_rate", 0.005)`

## Step 9: Expand Report Content
**File:** `latex/report.tex`
**Changes:**
- Expand Technical Challenges section with more details
- Add explicit Future Work section with concrete next steps
- Add system architecture section with diagram reference

## Step 10: Minor Improvements
- Add ROC curve to the save_all_plots registry
- Update README.md to document new features
- Add ROC curve entry to results/plots/README.md
- Update generate_latex_tables.py with significance results

---

## Implementation Order

1. Step 8 (GCN defaults) — quick code fix, no dependencies
2. Step 1 (Student ID) — quick fix
3. Step 7 (ROC curves) — new plot function
4. Step 6 (Statistical tests) — new evaluation function
5. Step 2 (Dataset Manager UI) — UI reorganization
6. Step 3 (Run All Methods) — new UI feature
7. Step 9 (Expand report) — LaTeX content
8. Step 4 (Architecture diagram) — LaTeX content
9. Step 5 (Screenshots) — requires running the app
10. Step 10 (Minor improvements) — documentation
