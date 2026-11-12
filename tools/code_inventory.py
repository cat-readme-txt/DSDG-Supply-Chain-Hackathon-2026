"""Write or verify the complete Python-file index in the root README.

Every direct scored generator is named explicitly. Supporting package files
inherit the mapping of their frozen source snapshot. A new top-level research
file must be classified here before the README check can pass.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BEGIN = "<!-- BEGIN PYTHON FILE INVENTORY -->"
END = "<!-- END PYTHON FILE INVENTORY -->"

DIRECT = {
    "pipeline.py": ("Phase 1", "44.95, 45.80, 46.21, 46.22"),
    "methods/policy_v1_54_10/build_forecast_hybrid.py": ("Phase 1", "50.44"),
    "experiments/dual_hazard_three_lane.py": ("Phase 1", "66.98"),
    "experiments/lithium_share_only.py": ("Phase 1", "67.26"),
    "experiments/phase2_public_oracle.py": ("Phase 2", "84.27, 89.45"),
    "experiments/public_diversification_qp.py": ("Phase 2", "91.36, 92.31"),
    "experiments/public_joint_order_search.py": ("Phase 2", "92.62 (near reproduction)"),
    "experiments/private_closure_conditional.py": ("Phase 2", "selected final pair; private score unknown"),
    "experiments/private_closure_backtest.py": ("Phase 2", "selected final pair evidence"),
    "experiments/private_closure_pair_backtest.py": ("Phase 2", "selected final pair evidence"),
    "methods/sk4_phase2/eda/33_fill_public.py": ("Phase 2", "SK4 parents for selected final pair"),
    "methods/check_provenance.py": ("Both", "audit: 16 graded files and selected final pair"),
    "experiments/submission_audit.py": ("Both", "audit: all stored CSVs"),
    "tools/code_inventory.py": ("Both", "documentation inventory"),
    "kaggle/build_notebook.py": ("Phase 1", "platform packaging; no direct score"),
    "databricks/track_notebook.py": ("Both", "platform packaging; no direct score"),
    "tests/test_pipeline.py": ("Phase 1", "tests for 44.95–46.22 source"),
}

PHASE2_RESEARCH = {
    "correlated_brier.py", "diversification_metric_inference.py",
    "event_calendar.py", "event_saa_backtest.py", "event_saa_policy.py",
    "joint_surface_sweep.py", "phase2_metric_check.py",
    "public_material_portfolio.py", "public_multistart.py",
    "public_qp_scale_sweep.py", "score_surface_fit.py",
}
PHASE1_RESEARCH = {
    "allocation_sweep.py", "brier_shrink_oof.py",
    "correlated_event_scenarios.py", "correlated_lane_buffer.py",
    "correlated_lane_buffer_candidate.py", "cvar_backtest.py", "cvar_lp.py",
    "diversification_frontier.py", "graded_comparison.py",
    "insured_pair_check.py", "leaderboard_inverse.py",
    "lean_share_sweep.py", "lithium_share_scenarios.py",
    "mape_price_model.py", "mape_threshold_sweep.py",
    "material_policy_selector.py", "private_risk_hedge.py",
    "ridge_forecast.py", "robust_lp_sweep.py", "scenario_planner.py",
}


def paths() -> list[str]:
    result = subprocess.run(["rg", "--files", "-g", "*.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(result.stdout.splitlines())


def classification(path: str) -> tuple[str, str]:
    if path in DIRECT:
        return DIRECT[path]
    if path.startswith("methods/policy_v1_54_10/src/"):
        return "Phase 1", "shared source: 54.10; 50.44 forecasts"
    if path.startswith("methods/lean_60_66/src/"):
        return "Phase 1", "shared source: 48.55, 60.66"
    if path.startswith("methods/lean_split_67_74/src/"):
        return "Phase 1; reused in 2", "shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62"
    if path.startswith("methods/lean_split_67_74/"):
        return "Phase 1", "research/test for 67.74; no direct CSV"
    if path.startswith("methods/sk4_phase2/src/"):
        return "Phase 2", "SK4 parents (public 89.45, noncanonical); selected final pair"
    if path.startswith("methods/sk4_phase2/"):
        return "Phase 2", "research/test for selected final pair"
    if path.startswith("scratch/"):
        return "Phase 1", "rejected research; no scored CSV"
    if path.startswith("experiments/"):
        name = Path(path).name
        if name in PHASE1_RESEARCH:
            return "Phase 1", "research; no direct scored CSV"
        if name in PHASE2_RESEARCH:
            return "Phase 2", "research; no direct scored CSV"
    raise ValueError(f"Unclassified Python file: {path}")


def role(path: str) -> str:
    source = (ROOT / path).read_text()
    doc = ast.get_docstring(ast.parse(source)) or ""
    first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    if not first:
        first = "Package marker" if Path(path).name == "__init__.py" else "Support module"
    return first.replace("|", "\\|")[:105]


def inventory() -> str:
    all_paths = paths()
    rows = [f"Python files indexed: **{len(all_paths)}**. Score shorthand `44.95` means "
            "`submissions/submission44.95.csv`; `unscored` means no graded score.\n"]
    groups = ("Phase 1", "Phase 1; reused in 2", "Phase 2", "Both")
    tagged = [(path, *classification(path)) for path in all_paths]
    for group in groups:
        matching = [(path, mapping) for path, phase, mapping in tagged if phase == group]
        rows.extend([f"### {group} ({len(matching)} files)", "",
                     "| Python file | Score mapping / artifact | Role |",
                     "| --- | --- | --- |"])
        for path, mapping in matching:
            rows.append(f"| [`{path}`]({path}) | {mapping} | {role(path)} |")
        rows.append("")
    assert sum(1 for _, phase, _ in tagged if phase in groups) == len(all_paths)
    return "\n".join(rows).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace the README inventory")
    args = parser.parse_args()
    document = README.read_text()
    before, tail = document.split(BEGIN, 1)
    _, after = tail.split(END, 1)
    new = before + BEGIN + "\n" + inventory() + END + after
    if args.write:
        README.write_text(new)
        print(f"Wrote {len(paths())} Python-file mappings to {README}")
    elif document != new:
        raise SystemExit("README Python-file inventory is stale; run python tools/code_inventory.py --write")
    else:
        print(f"README covers all {len(paths())} Python files")


if __name__ == "__main__":
    main()
