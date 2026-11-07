# Databricks track runbook

This is the Databricks version of the same source pipeline that generated the
local candidates. The notebook creates actual Delta tables, Spark SQL features,
an MLflow experiment run, and dashboard source tables. The AI cell first tries
the built-in `ai_classify` function; a custom `ai_query` endpoint is optional.
**Creating these
files locally does not count as using a Databricks feature; run them in the
team's workspace and retain the run and dashboard links as evidence.**

1. In your **Databricks Free Edition workspace**, use **New > Add or upload
   data > Upload files to a volume**. Create or select a volume, make a `data`
   folder, and upload all ten `data/*.csv` files into that folder. Import
   `pipeline.py` into a Workspace folder. A Kaggle notebook is a separate
   runtime and does not execute this Databricks track workflow.
2. Import `track_notebook.py` as a Python notebook. Set `project_root` to the
   Workspace folder with `pipeline.py`, `data_root` to the volume folder with
   the CSVs, and `artifact_root` to a writable volume folder.
3. Run all cells. Confirm that it reports 260 candidate rows, 30 supplier rows,
   12 historical folds for Phase 1, a 260-row `dsdg_candidate` table, and an
   MLflow run ID. The notebook independently rebuilds the preferred 260-row
   submission with Spark SQL and stops if any field differs from the local
   generator. Download `databricks_news_primary_w208.csv` from the volume.
   For Phase 1, its expected SHA-256 is
   `9683650ddd2477dc38fff05cbd7b70540884087e2bf9e239c892db710d003d5b`.
4. Run the AI cell. It classifies up to 20 recent released news headlines with
   `ai_classify` and records the count in MLflow. If you prefer a supported
   model endpoint, set `ai_endpoint` to its exact name to use `ai_query`.
   If the workspace cannot run either AI function, retain the printed error;
   the AI form selection remains unverified. The AI output is a review table
   and does not change the submission without a separate backtest.
5. Run the queries in `dashboard_queries.sql`, create an AI/BI dashboard with
   the metric, weekly risk, and high-risk material views, and save its link.
   Only run the optional AI query after the AI cell succeeds.

Record the actual Databricks workspace, MLflow run ID, feature row count,
candidate hash, AI endpoint/result count, and dashboard link in `PROGRESS.md`.
Until those steps run, the form selections describe intended work, not verified
platform use. The notebook does not upload to Kaggle or submit the opt-in form.

The notebook uses a table prefix `dsdg_` in the current catalog/schema. Use a
team-owned schema because reruns overwrite these tables. It needs permission to
create tables and write to the chosen volume. AI functions need a supported
workspace region and compute environment; a supplied `ai_query` endpoint must
also be available in the workspace.

Databricks references: [Free Edition](https://docs.databricks.com/gcp/en/getting-started/free-edition),
[volume upload](https://docs.databricks.com/gcp/en/ingestion/file-upload/),
[Workspace Python files](https://docs.databricks.com/aws/en/files/workspace-modules),
[MLflow tracking](https://docs.databricks.com/gcp/en/mlflow/tracking),
[`ai_classify` requirements](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify),
[`ai_query` requirements](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query),
and [notebook dashboards](https://docs.databricks.com/aws/en/notebooks/dashboards).
