"""Build an offline, self-contained Kaggle notebook from the local pipeline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline.py"
OUTPUT = Path(__file__).with_name("runner.ipynb")

pipeline_source = PIPELINE.read_text(encoding="utf-8")
entrypoint = '\nif __name__ == "__main__":\n'
if pipeline_source.count(entrypoint) != 1:
    raise ValueError("Could not isolate pipeline definitions from the CLI entrypoint")
pipeline_definitions = pipeline_source.split(entrypoint, 1)[0].rstrip() + "\n"

setup_source = '''from pathlib import Path
import os

REQUIRED_CSVS = {
    "news.csv", "prices.csv", "tariffs.csv", "shipping.csv", "macro.csv",
    "suppliers.csv", "demand.csv", "outcomes.csv", "supply_history.csv",
    "sample_submission.csv",
}

# Set this to the exact directory if more than one data version is attached.
DATA_DIR = Path(os.environ["DSDG_DATA_DIR"]) if os.environ.get("DSDG_DATA_DIR") else None
if DATA_DIR is None:
    candidate_dirs = sorted({path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv")})
    valid_dirs = [path for path in candidate_dirs if all((path / name).is_file() for name in REQUIRED_CSVS)]
    if len(valid_dirs) != 1:
        raise RuntimeError(
            f"Expected one folder with all ten CSVs, found {valid_dirs}. "
            "Set DATA_DIR = Path('/kaggle/input/YOUR_DATA_FOLDER') in this cell."
        )
    DATA_DIR = valid_dirs[0]
missing = sorted(name for name in REQUIRED_CSVS if not (DATA_DIR / name).is_file())
if missing:
    raise FileNotFoundError(f"Missing files in {DATA_DIR}: {missing}")

WORK_DIR = Path(os.environ.get("DSDG_OUTPUT_DIR", "/kaggle/working"))
SUBMISSION_DIR = WORK_DIR / "submissions"
REPORT_DIR = WORK_DIR / "reports"
print("Competition data:", DATA_DIR)
print("Output folder:", WORK_DIR)
'''

run_source = '''import hashlib
import json

data = CompetitionData(DATA_DIR)
audit = data.audit()
forecast = data.forecast_backtest()
search = data.policy_search()
simulator = {name: data.simulator_backtest(profile) for name, profile in PROFILES.items()}

REPORT_DIR.mkdir(parents=True, exist_ok=True)
backtest_path = REPORT_DIR / f"backtest_w{data.historical_end}.json"
backtest_path.write_text(json.dumps({
    "data_week": data.historical_end,
    "forecast": forecast,
    "simulator": simulator,
    "source": "Offline Kaggle notebook generated from local pipeline.py",
}, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
search_path = REPORT_DIR / f"policy_search_w{data.historical_end}.json"
search_path.write_text(json.dumps(search, indent=2, allow_nan=False) + "\\n", encoding="utf-8")

submissions = {
    name: data.write_candidate(PROFILES[name], SUBMISSION_DIR)
    for name in SUBMISSION_PROFILE_NAMES
}
for name, path in submissions.items():
    data.validate_file(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {path} | rows=260 | sha256={digest}")
print("historical_end_week:", data.historical_end)
print("forecast_folds:", forecast["fold_count"])
print("forecast_summary:", forecast["summary"])
print("policy_trials:", search["trial_count"])
print("backtest_report:", backtest_path)
print("policy_search_report:", search_path)
'''


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(True),
    }


notebook = {
    "cells": [
        markdown(
            "# DSDG supply-chain Kaggle runner\n\n"
            "This notebook is built from the local `pipeline.py` and uses only Python's "
            "standard library. Attach the ten competition CSVs as notebook input. "
            "It generates local Kaggle submission files; it does not run Databricks "
            "Delta, MLflow, AI functions, or dashboards.\n"
        ),
        code(pipeline_definitions),
        code(setup_source),
        code(run_source),
        markdown(
            "The validated CSVs are in `/kaggle/working/submissions/`. Download the "
            "chosen file from notebook outputs. Running this notebook does not submit "
            "it to the competition. The state-only `primary` candidate already scored "
            "44.95 twice; `news_primary` is a different, unsubmitted comparison.\n"
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(notebook["cells"], start=1):
    cell["id"] = f"dsdg-{index}"
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
