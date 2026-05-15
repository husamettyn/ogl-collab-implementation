#!/usr/bin/env python3
"""Grade all 6 runs for iteration-1 of parallel-exploring skill."""
import json, os, re

WORKSPACE = "/home/husam/Desktop/YTU-YL/webmining/ogl-collab-implementation/parallel-exploring-workspace/iteration-1"

EVALS = {
    "eval-1-ui-structure": {
        "assertions": [
            "Report mentions src/ui/app.py as the main entry point",
            "Report mentions src/ui/pages.py for navigation/pages module",
            "Report identifies at least 4 distinct UI page files",
            "Report explains how page navigation/registration works",
            "Every finding includes a file path and line number citation",
        ]
    },
    "eval-2-ml-methods": {
        "assertions": [
            "Report mentions ML method files under src/methods/",
            "Report describes what each method does (not just listing names)",
            "Every finding includes a file path and line number citation",
            "Report references at least 3 distinct method implementation files",
        ]
    },
    "eval-3-data-flow": {
        "assertions": [
            "Report covers the data loading stage (mentions src/data/ or data loading code)",
            "Report covers the training pipeline (mentions src/train.py or training code)",
            "Report covers the evaluation stage (mentions src/evaluation/ or evaluation code)",
            "Report connects the stages into a coherent end-to-end flow",
            "Every finding includes a file path and line number citation",
        ]
    },
}

CONFIGS = ["with_skill", "without_skill"]

def read_report(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""

def grade_eval(eval_name, assertions, report_path, timing_path):
    text = read_report(report_path).lower()

    results = []
    for assertion in assertions:
        a = assertion.lower()
        passed = False
        evidence = ""

        if eval_name == "eval-1-ui-structure":
            if "src/ui/app.py" in a and "entry point" in a:
                passed = "src/ui/app.py" in text and ("main" in text or "entry" in text)
                evidence = "Found app.py references" if passed else "app.py not mentioned as entry point"
            elif "src/ui/pages.py" in a:
                passed = "pages.py" in text or "src/ui/pages" in text
                evidence = "Found pages.py references" if passed else "pages.py not mentioned"
            elif "at least 4" in a and "page" in a:
                page_files = re.findall(r'page_(\w+)\.py', text)
                passed = len(set(page_files)) >= 4
                evidence = f"Found {len(set(page_files))} distinct page files: {set(page_files)}"
            elif "navigation" in a or "registration" in a:
                passed = ("st.tabs" in text or "navigation" in text or "tab" in text) and ("render" in text)
                evidence = "Navigation mechanism explained" if passed else "Navigation not explained"
            elif "line number" in a or "citation" in a:
                line_refs = len(re.findall(r'(line|satır)\s*\d+', text))
                file_refs = len(re.findall(r'```|\.py:\d+|\.py\b.*\blines?\s*\d+', text))
                passed = line_refs >= 5 or file_refs >= 5
                evidence = f"Found ~{line_refs} line number references, ~{file_refs} file path patterns"

        elif eval_name == "eval-2-ml-methods":
            if "src/methods/" in a:
                passed = "src/methods/" in text or "methods/" in text
                evidence = "Methods directory referenced" if passed else "No methods/ reference"
            elif "describes what each method does" in a:
                has_cn = "common" in text and ("neighbor" in text or "heuristic" in text)
                has_mlp = "mlp" in text and ("feature" in text or "perceptron" in text or "neural" in text)
                has_gcn = "gcn" in text and ("graph" in text or "convolution" in text or "message" in text)
                passed = has_cn and has_mlp and has_gcn
                evidence = f"CN described: {has_cn}, MLP described: {has_mlp}, GCN described: {has_gcn}"
            elif "line number" in a or "citation" in a:
                line_refs = len(re.findall(r'(line|satır)\s*\d+', text))
                file_refs = len(re.findall(r'\.py:\d|\.py\b.*\blines?\s*\d', text))
                passed = line_refs >= 5 or file_refs >= 5
                evidence = f"Found ~{line_refs} line refs, ~{file_refs} file patterns"
            elif "at least 3" in a and "method" in a:
                methods = set(re.findall(r'(common_neighbors|mlp|gcn)', text))
                passed = len(methods) >= 3
                evidence = f"Found {len(methods)} distinct methods: {methods}"

        elif eval_name == "eval-3-data-flow":
            if "data loading" in a or "src/data/" in a:
                passed = ("src/data/" in text or "loader" in text or "load_collab" in text.lower() or "dataset" in text)
                evidence = "Data loading covered" if passed else "Data loading not covered"
            elif "training" in a or "src/train.py" in a:
                passed = ("train" in text and ("epoch" in text or "model" in text or "optimizer" in text or "run_" in text or "fit" in text))
                evidence = "Training pipeline covered" if passed else "Training not covered"
            elif "evaluation" in a or "src/evaluation/" in a:
                passed = ("evaluat" in text or "metrics" in text or "hits@" in text or "hits_at" in text)
                evidence = "Evaluation covered" if passed else "Evaluation not covered"
            elif "end-to-end" in a or "connect" in a or "coherent" in a:
                passed = ("flow" in text or "pipeline" in text or "stage" in text or "end-to-end" in text or "end to end" in text)
                evidence = "End-to-end flow described" if passed else "No coherent flow"
            elif "line number" in a or "citation" in a:
                line_refs = len(re.findall(r'(line|satır)\s*\d+', text))
                file_refs = len(re.findall(r'\.py:\d|\.py\b.*\blines?\s*\d', text))
                passed = line_refs >= 5 or file_refs >= 5
                evidence = f"Found ~{line_refs} line refs, ~{file_refs} file patterns"

        results.append({
            "text": assertion,
            "passed": passed,
            "evidence": evidence
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    timing = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)

    grading = {
        "expectations": results,
        "summary": {
            "passed": passed_count,
            "failed": total - passed_count,
            "total": total,
            "pass_rate": round(passed_count / total, 2) if total > 0 else 0.0
        },
        "timing": {
            "executor_duration_seconds": timing.get("total_duration_seconds", 0),
            "total_duration_seconds": timing.get("total_duration_seconds", 0),
        },
        "claims": [],
        "user_notes_summary": {"uncertainties": [], "needs_review": [], "workarounds": []},
        "eval_feedback": {"suggestions": [], "overall": "No suggestions"}
    }

    return grading

for eval_name, info in EVALS.items():
    for config in CONFIGS:
        run_dir = os.path.join(WORKSPACE, eval_name, config)
        report_path = os.path.join(run_dir, "outputs", "report.md")
        timing_path = os.path.join(run_dir, "timing.json")
        grading_path = os.path.join(run_dir, "grading.json")

        print(f"\n{'='*60}")
        print(f"Grading {eval_name} / {config}")
        print(f"Report exists: {os.path.exists(report_path)}")

        grading = grade_eval(eval_name, info["assertions"], report_path, timing_path)

        with open(grading_path, "w") as f:
            json.dump(grading, f, indent=2)

        summary = grading["summary"]
        print(f"Result: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.0%})")
        for r in grading["expectations"]:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['text'][:80]}...")

print("\n\nDone! All grading complete.")
