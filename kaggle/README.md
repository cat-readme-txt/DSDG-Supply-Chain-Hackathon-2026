# Kaggle notebook fallback

The Kaggle Code notebook and the Databricks Free Edition workspace are separate
systems. `databricks/track_notebook.py` needs Databricks `dbutils`, Spark/Delta,
MLflow, Unity Catalog, and AI functions, so it cannot run as-is in Kaggle. The
`mlflow` import error is the first mismatch. Installing MLflow in Kaggle would
not provide the other Databricks services, and PyPI is unreachable in the
reported notebook session.

Use [runner.ipynb](runner.ipynb) in Kaggle instead:

1. Import the notebook in Kaggle Code and attach the ten competition CSVs as
   input. The notebook searches `/kaggle/input` for a folder containing all ten.
2. Run all cells. No `%pip` or internet access is needed. If it finds multiple
   data versions, set `DATA_DIR` in the setup cell to the intended folder.
3. Download the validated CSVs from `/kaggle/working/submissions/`. Phase 1
   files end in `_w208.csv`; after Phase 2 data is attached, files end in
   `_w221.csv`. Compare the printed SHA-256 with the local copy if you need
   exact parity. The notebook does not upload a submission automatically.

The notebook embeds a snapshot of `pipeline.py`; the local file stays intact.
After any local pipeline change, regenerate the notebook with
`python3 kaggle/build_notebook.py` before importing it again.

This Kaggle runner is only a submission fallback. The Databricks track form's
five feature selections require the separate [Databricks runbook](../databricks/README.md)
to be executed in a Databricks workspace.
