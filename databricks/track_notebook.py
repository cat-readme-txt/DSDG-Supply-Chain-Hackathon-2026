# Databricks notebook source
"""DSDG supply-chain track: native data, features, tracking, and submission run.

Import this file as a Databricks Python notebook. The source pipeline.py and all
ten competition CSVs must be uploaded before running the notebook.
"""

# COMMAND ----------

import importlib.util
import csv
import json
import re
from pathlib import Path

widget_defaults = {
    "project_root": "/Workspace/Users/YOU/dsdg",
    "data_root": "/Volumes/dsdg-hackathon/default/data/data",
    "artifact_root": "/Volumes/dsdg-hackathon/default/data/artifacts",
    "catalog": "dsdg-hackathon",
    "schema": "default",
    "ai_endpoint": "",
}
existing_widgets = dbutils.widgets.getAll()
for name, default in widget_defaults.items():
    if name not in existing_widgets:
        dbutils.widgets.text(name, default)

project_root = Path(dbutils.widgets.get("project_root"))
data_root = Path(dbutils.widgets.get("data_root"))
artifact_root = Path(dbutils.widgets.get("artifact_root"))
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
ai_endpoint = dbutils.widgets.get("ai_endpoint").strip()

try:
    import mlflow
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "MLflow is missing from this Databricks notebook environment. "
        "Choose the ML base environment in the Environment pane if available, "
        "or add the mlflow package there, then rerun."
    ) from exc

for label, value in (("catalog", catalog), ("schema", schema)):
    if value and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid {label} widget: {value!r}")
if catalog.lower() in {"catalog", "data", "my_catalog"}:
    raise ValueError(
        f"catalog widget is {catalog!r}; this workspace's catalog is 'dsdg-hackathon'. "
        "Edit the existing UI parameter, then rerun the cell."
    )
if catalog:
    spark.sql(f"USE CATALOG `{catalog}`")
if schema:
    spark.sql(f"USE SCHEMA `{schema}`")
print("Delta table namespace:", spark.sql("SELECT current_catalog(), current_schema()").first())

if not (project_root / "pipeline.py").is_file():
    raise FileNotFoundError(f"Upload pipeline.py to {project_root} and set project_root")
if not (data_root / "sample_submission.csv").is_file():
    raise FileNotFoundError(f"Upload all ten CSVs to {data_root} and set data_root")
artifact_root.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("dsdg_pipeline", project_root / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)

# COMMAND ----------

# Delta tables contain the actual competition inputs used by the run. The
# source CSVs are retained so pipeline.py can generate the identical candidates.
csv_names = (
    "news", "prices", "tariffs", "shipping", "macro", "suppliers",
    "demand", "outcomes", "supply_history", "sample_submission",
)
row_counts = {}
for name in csv_names:
    source = data_root / f"{name}.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = spark.read.option("header", "true").option("inferSchema", "true").csv(str(source))
    row_counts[name] = frame.count()
    frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"dsdg_raw_{name}")

assert row_counts["sample_submission"] == 260, row_counts
assert row_counts["suppliers"] == 30, row_counts
print("Delta ingestion complete:", row_counts)

# COMMAND ----------

# Spark SQL derives real rolling price, disruption, and headline-warning
# features from released weeks. These are inspectable in the Delta tables.
news_features = spark.sql("""
    SELECT n.week, s.material,
           COUNT(DISTINCT n.article_id) AS warning_headlines
    FROM dsdg_raw_news n
    CROSS JOIN (SELECT DISTINCT material FROM dsdg_raw_suppliers) s
    WHERE lower(n.headline) LIKE concat('%', replace(s.material, '_', ' '), '%')
      AND lower(n.headline) RLIKE
          'export curbs|strike date|strike on|sanctions vote|drought|shortage|closure|halt exports|dockworkers|congestion builds|tensions rise|labor talks drag|slower license approvals'
      AND lower(n.headline) NOT RLIKE 'averted|deal reached|calm returns|eases|returns to'
    GROUP BY n.week, s.material
""")
news_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dsdg_news_features")

features = spark.sql("""
    WITH rolling AS (
      SELECT o.week, o.material, o.benchmark_price, o.disrupted,
             AVG(o.benchmark_price) OVER (
               PARTITION BY o.material ORDER BY o.week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
             ) AS price_mean_4w,
             AVG(o.disrupted) OVER (
               PARTITION BY o.material ORDER BY o.week ROWS BETWEEN 51 PRECEDING AND CURRENT ROW
             ) AS disruption_rate_52w,
             LAG(o.benchmark_price) OVER (PARTITION BY o.material ORDER BY o.week) AS prior_price
      FROM dsdg_raw_outcomes o
    )
    SELECT r.*, COALESCE(n.warning_headlines, 0) AS warning_headlines
    FROM rolling r LEFT JOIN dsdg_news_features n
      ON r.week = n.week AND r.material = n.material
""")
features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dsdg_features")
feature_count = spark.table("dsdg_features").count()
assert feature_count == row_counts["outcomes"], (feature_count, row_counts["outcomes"])
print("Spark SQL feature rows:", feature_count)

# COMMAND ----------

# Run the same leak-free historical tests and generator used for the local
# submission. Delta/SQL features above make the Databricks data path auditable;
# only the tested pipeline.py policy determines the CSV uploaded to Kaggle.
data = pipeline.CompetitionData(data_root)
audit = data.audit()
forecast = data.forecast_backtest()
policy_search = data.policy_search()
submission_dir = artifact_root / "submissions"
submission_paths = {
    name: data.write_candidate(pipeline.PROFILES[name], submission_dir)
    for name in pipeline.SUBMISSION_PROFILE_NAMES
}

# Rebuild the preferred submission natively with Spark SQL. The local Python
# generator is an independent parity check, not the source for this file.
origin_rows = [
    (start, end, min(data.historical_end, start - 1))
    for start, end in pipeline.WINDOWS
]
spark.createDataFrame(origin_rows, ["start_week", "end_week", "origin"]) \
    .createOrReplaceTempView("dsdg_submission_origins")
warning_rows = [
    (origin, material, float(data.warning_score(material, origin)))
    for origin in sorted({row[2] for row in origin_rows})
    for material in data.materials
]
spark.createDataFrame(warning_rows, ["origin", "material", "warning_score"]) \
    .createOrReplaceTempView("dsdg_submission_warnings")

native = spark.sql("""
    WITH sample AS (
      SELECT id,
             CAST(regexp_extract(id, '^w([0-9]+)_', 1) AS INT) AS week,
             regexp_extract(id, '^w[0-9]+_(.*)$', 1) AS material
      FROM dsdg_raw_sample_submission
    ),
    origins AS (SELECT DISTINCT origin FROM dsdg_submission_origins),
    stats AS (
      SELECT x.origin, o.material,
             AVG(CAST(o.disrupted AS DOUBLE)) AS historical_rate,
             MAX_BY(CAST(o.disrupted AS DOUBLE), o.week) AS last_disrupted,
             MAX_BY(CAST(o.benchmark_price AS DOUBLE), o.week) AS last_price,
             AVG(CASE WHEN o.week > x.origin - 4
                      THEN CAST(o.benchmark_price AS DOUBLE) END) AS recent_price
      FROM origins x JOIN dsdg_raw_outcomes o ON o.week <= x.origin
      GROUP BY x.origin, o.material
    ),
    recent_supplier_cost AS (
      SELECT x.origin, h.material, h.country,
             AVG(CAST(h.landed_cost AS DOUBLE)) AS mean_landed_cost
      FROM origins x JOIN dsdg_raw_supply_history h
        ON h.week BETWEEN x.origin - 25 AND x.origin
      GROUP BY x.origin, h.material, h.country
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY origin, material ORDER BY mean_landed_cost, country
      ) AS supplier_rank
      FROM recent_supplier_cost
    ),
    allocation AS (
      SELECT origin, material,
             SUM(CASE WHEN country = 'KES' THEN share ELSE 0 END) AS share_KES,
             SUM(CASE WHEN country = 'VAL' THEN share ELSE 0 END) AS share_VAL,
             SUM(CASE WHEN country = 'SUN' THEN share ELSE 0 END) AS share_SUN,
             SUM(CASE WHEN country = 'ORI' THEN share ELSE 0 END) AS share_ORI,
             SUM(CASE WHEN country = 'HAL' THEN share ELSE 0 END) AS share_HAL,
             SUM(CASE WHEN country = 'AUR' THEN share ELSE 0 END) AS share_AUR
      FROM (
        SELECT origin, material, country,
               CASE WHEN supplier_rank = 3 THEN 0.333334 ELSE 0.333333 END AS share
        FROM ranked
      ) x
      GROUP BY origin, material
    ),
    joined AS (
      SELECT s.id, s.week, s.material, w.start_week, w.origin,
             CAST(d.demand_units AS DOUBLE) AS demand_units,
             t.historical_rate, t.last_disrupted, t.last_price, t.recent_price,
             COALESCE(n.warning_score, 0.0) AS warning_score,
             a.share_KES, a.share_VAL, a.share_SUN,
             a.share_ORI, a.share_HAL, a.share_AUR
      FROM sample s JOIN dsdg_submission_origins w
        ON s.week BETWEEN w.start_week AND w.end_week
      JOIN dsdg_raw_demand d ON d.week = s.week AND d.material = s.material
      JOIN stats t ON t.origin = w.origin AND t.material = s.material
      JOIN allocation a ON a.origin = w.origin AND a.material = s.material
      LEFT JOIN dsdg_submission_warnings n
        ON n.origin = w.origin AND n.material = s.material
    ),
    state AS (
      SELECT *, historical_rate +
               (last_disrupted - historical_rate) * POW(0.8, week - origin) AS state_probability
      FROM joined
    )
    SELECT id,
           LEAST(1.0, GREATEST(0.0,
             state_probability + (1 - state_probability) * 0.6 *
             warning_score * POW(0.95, week - origin))) AS disruption_prob,
           GREATEST(0.0001, last_price + 0.25 * (recent_price - last_price)) AS price_forecast,
           CASE WHEN week - start_week < 8
                THEN CAST(BROUND(demand_units * 1.2, 0) AS BIGINT) ELSE 0 END AS order_units,
           share_KES, share_VAL, share_SUN, share_ORI, share_HAL, share_AUR
    FROM state
""")
assert native.count() == 260
native_by_id = {row["id"]: row.asDict() for row in native.collect()}
assert set(native_by_id) == {row["id"] for row in data.sample}

native_path = submission_dir / f"databricks_news_primary_w{data.historical_end}.csv"
with native_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=data.sample_header)
    writer.writeheader()
    for template in data.sample:
        source = native_by_id[template["id"]]
        row = {
            "id": template["id"],
            "disruption_prob": f"{source['disruption_prob']:.6f}",
            "price_forecast": f"{source['price_forecast']:.4f}",
            "order_units": str(source["order_units"]),
        }
        for column in data.share_columns:
            # The local generator writes ineligible suppliers as literal "0".
            # Preserve that exact CSV representation for byte-level parity.
            row[column] = "0" if source[column] == 0 else f"{source[column]:.6f}"
        writer.writerow(row)
data.validate_file(native_path)

_, native_rows = pipeline.read_csv(native_path)
_, local_rows = pipeline.read_csv(submission_paths["news_primary"])
parity_differences = [
    (a["id"], column, a[column], b[column])
    for a, b in zip(native_rows, local_rows)
    for column in data.sample_header
    if a[column] != b[column]
]
if parity_differences:
    raise AssertionError(f"Spark/local submission mismatch: {parity_differences[:10]}")
print("Spark SQL candidate matches local generator exactly:", native_path)

report_path = artifact_root / f"databricks_run_w{data.historical_end}.json"
report = {
    "historical_end": data.historical_end,
    "source_rows": row_counts,
    "spark_feature_rows": feature_count,
    "forecast": forecast,
    "policy_search": policy_search,
    "submissions": {name: str(path) for name, path in submission_paths.items()},
    "spark_sql_candidate": str(native_path),
    "spark_sql_candidate_parity_differences": len(parity_differences),
}
report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

summary = forecast["summary"]
metrics = {
    "brier_historical_rate": summary["brier_full"],
    "brier_state_decay": summary["brier_state_decay"],
    "brier_news_decay": summary["brier_news_decay"],
    "price_mape_last": summary["mape_last"],
    "price_mape_blend": summary["mape_price_blend"],
    "forecast_fold_count": forecast["fold_count"],
    "sql_feature_rows": feature_count,
    "spark_candidate_parity_differences": len(parity_differences),
}
best = policy_search["top_eligible"][0]
metrics["selected_train_cost"] = best["training"]["cost"]
metrics["selected_train_shock_fill"] = best["training"]["shock_fill_rate"]
metrics["selected_holdout_cost"] = best["holdout"]["cost"]
metrics["selected_holdout_shock_fill"] = best["holdout"]["shock_fill_rate"]

with mlflow.start_run(run_name=f"dsdg_w{data.historical_end}_news_primary") as run:
    mlflow.log_param("historical_end_week", data.historical_end)
    mlflow.log_param("selected_shares", best["shares"])
    mlflow.log_param("selected_order_multiplier", best["safety"])
    mlflow.log_param("selected_stop_weeks", best["stop"])
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(report_path))
    for path in submission_paths.values():
        mlflow.log_artifact(str(path), artifact_path="submissions")
    mlflow.log_artifact(str(native_path), artifact_path="submissions")
    run_id = run.info.run_id

metric_rows = [(run_id, data.historical_end, key, float(value)) for key, value in metrics.items()]
spark.createDataFrame(metric_rows, ["run_id", "historical_end_week", "metric", "value"]) \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("dsdg_run_metrics")

candidate = spark.read.option("header", "true").option("inferSchema", "true").csv(
    str(native_path)
)
candidate.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dsdg_candidate")
assert spark.table("dsdg_candidate").count() == 260
print("MLflow run:", run_id)
print("Spark SQL Kaggle candidate:", native_path)

# COMMAND ----------

# AI news cross-check on released headlines only. ai_classify is the default;
# set ai_endpoint to use ai_query instead. The result does not alter the CSV.
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW dsdg_ai_input AS
    SELECT article_id, week, headline
    FROM dsdg_raw_news
    WHERE week >= {data.historical_end - 1}
      AND lower(headline) RLIKE
          'export|strike|sanction|drought|shortage|closure|congestion|tariff|policy'
    ORDER BY week DESC, article_id
    LIMIT 20
""")
if ai_endpoint and not re.fullmatch(r"[A-Za-z0-9_.-]+", ai_endpoint):
    raise ValueError("ai_endpoint contains invalid characters")
if ai_endpoint:
    ai_sql = f"""
        SELECT article_id, week, headline,
               ai_query('{ai_endpoint}',
                   concat('Classify this fictional supply-chain headline as likely disruption warning, routine, or resolved. Reply with one label only: ', headline)
               ) AS ai_label
        FROM dsdg_ai_input
    """
else:
    ai_sql = """
        SELECT article_id, week, headline,
               ai_classify(headline, ARRAY('likely disruption warning', 'routine', 'resolved rumor'))
                   AS ai_label
        FROM dsdg_ai_input
    """
try:
    spark.sql(ai_sql).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable("dsdg_ai_news_triage")
    ai_count = spark.table("dsdg_ai_news_triage").count()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("ai_triaged_headlines", ai_count)
    spark.createDataFrame(
        [(run_id, data.historical_end, "ai_triaged_headlines", float(ai_count))],
        ["run_id", "historical_end_week", "metric", "value"],
    ).write.format("delta").mode("append").saveAsTable("dsdg_run_metrics")
    print("AI-classified released headlines:", ai_count)
except Exception as exc:
    print("AI triage did not run in this workspace. Save this error for follow-up:", repr(exc))

# COMMAND ----------

display(spark.table("dsdg_run_metrics"))
display(spark.table("dsdg_candidate").limit(20))
