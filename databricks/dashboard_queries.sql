-- Run these in Databricks SQL and add the result visualizations to an AI/BI
-- dashboard. The tables are created by track_notebook.py in the current schema.

-- Forecast quality and local policy evidence. Use a bar chart by metric.
SELECT metric, value, historical_end_week, run_id
FROM dsdg_run_metrics
ORDER BY metric;

-- Submitted risk and order volume by week. Use a line chart by week.
SELECT CAST(regexp_extract(id, '^w([0-9]+)_', 1) AS INT) AS week,
       AVG(disruption_prob) AS mean_disruption_prob,
       SUM(order_units) AS total_order_units
FROM dsdg_candidate
GROUP BY CAST(regexp_extract(id, '^w([0-9]+)_', 1) AS INT)
ORDER BY week;

-- Highest-risk material weeks. Use a table.
SELECT id, disruption_prob, price_forecast, order_units
FROM dsdg_candidate
ORDER BY disruption_prob DESC
LIMIT 30;

-- Optional AI triage result, only after the ai_endpoint cell runs.
SELECT week, headline, ai_label
FROM dsdg_ai_news_triage
ORDER BY week DESC, headline;
