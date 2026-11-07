# Databricks track runbook

This is the Databricks version of the same source pipeline that generated the
local candidates. The notebook creates actual Delta tables, Spark SQL features,
an MLflow experiment run, and dashboard source tables. The AI cell first tries
the built-in `ai_classify` function; a custom `ai_query` endpoint is optional.
**Creating these
files locally does not count as using a Databricks feature; run them in the
team's workspace and retain the run and dashboard links as evidence.**

If a Kaggle notebook reports `ModuleNotFoundError: mlflow` at the first import,
you are in the Kaggle runtime. Use the separate [offline Kaggle runner](../kaggle/README.md)
for submission generation, then return here when you open a Databricks Free
Edition workspace. Installing MLflow in Kaggle will not add `dbutils`, Delta,
or Unity Catalog.

1. In your **Databricks Free Edition workspace**, use **New > Add or upload
   data > Upload files to a volume**. The screenshot confirms this hierarchy:
   catalog `dsdg-hackathon` → schema `default` → volume `data`. The first
   volume listing shows a `data/` subfolder. Confirm that all ten CSVs are
   inside it with `display(dbutils.fs.ls("/Volumes/dsdg-hackathon/default/data/data"))`.
   If they are, the input path is `/Volumes/dsdg-hackathon/default/data/data`. Import
   `pipeline.py` into a Workspace folder. A Kaggle notebook is a separate
   runtime and does not execute this Databricks track workflow.
2. Import `track_notebook.py` as a **Python notebook** in the same Workspace
   folder as the plain Python file `pipeline.py`. Attach serverless compute.
   Set `project_root` to that Workspace folder, `data_root` to the volume
   folder with the CSVs, and `artifact_root` to a writable volume folder.
   Set the `catalog` and `schema` widgets to a namespace where you can create
   tables. Leave `ai_endpoint` blank for the built-in classifier.
   The first Python cell creates any missing text widgets. If you already
   created parameters through **Edit > Add parameter**, the notebook keeps
   those values; set each widget type to **Text** and parameter type to
   **String**.

   Confirmed catalog/schema values: `catalog=dsdg-hackathon`, `schema=default`.
   If the CSVs appear in the listed subfolder, set
   `data_root=/Volumes/dsdg-hackathon/default/data/data`, and
   `artifact_root=/Volumes/dsdg-hackathon/default/data/artifacts`. Set
   `project_root=/Workspace/Users/YOUR_EMAIL/dsdg` when that is the folder
   containing `pipeline.py`. If you instead upload the CSVs directly to the
   volume root, use `data_root=/Volumes/dsdg-hackathon/default/data`.
   If the first cell reports `Catalog 'catalog' was not found`, edit your
   existing **catalog** UI parameter from the literal placeholder `catalog`
   to `dsdg-hackathon`; notebook defaults do not overwrite UI values.
   If it prints `dsdg-hackathon.default` and then reports that `pipeline.py`
   is missing, import the local [`pipeline.py`](../pipeline.py) as a plain
   Workspace **file** into the folder named by `project_root` (currently
   `/Workspace/Users/<your-account>/dsdg`). Check the exact file with
   `Path(project_root, "pipeline.py").is_file()` before rerunning. Pasting the
   source into another notebook cell does not create that Workspace file.
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

If `import mlflow` fails **inside Databricks**, choose an ML base environment
in the notebook's Environment pane if offered, or add the `mlflow` package to
that notebook environment, then rerun. Databricks provides the tracking server,
but the Python client package depends on the selected notebook environment.
If the error occurs in Kaggle instead, use the Kaggle fallback above.

Databricks references: [Free Edition](https://docs.databricks.com/gcp/en/getting-started/free-edition),
[volume upload](https://docs.databricks.com/gcp/en/ingestion/file-upload/),
[Workspace Python files](https://docs.databricks.com/aws/en/files/workspace-modules),
[MLflow tracking](https://docs.databricks.com/gcp/en/mlflow/tracking),
[`ai_classify` requirements](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify),
[`ai_query` requirements](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query),
and [notebook dashboards](https://docs.databricks.com/aws/en/notebooks/dashboards).
