#!/usr/bin/env bash
# Phase 2 runbook (Sat Nov 7, 12 PM CT): fresh data -> metric check -> retrain -> final pair.
# usage: bash eda/phase2.sh            (run from the repo root)
set -euo pipefail
C=dsdg-hackathon-2026-supply-chain-disruptor
set -a; . ./.env; set +a

echo "== 1. download (Phase 1 copy kept in data/comp_p1)"
[ -d data/comp_p1 ] || cp -r data/comp data/comp_p1
mkdir -p data/raw_p2
uv run kaggle competitions download -c $C -p data/raw_p2 --force
unzip -o data/raw_p2/*.zip -d data/comp
uv run python -c "
import pandas as pd
o = pd.read_csv('data/comp/outcomes.csv'); print('last released week:', o.week.max())
assert o.week.max() >= 221, 'Phase 2 data not out yet'
a, b = pd.read_csv('data/comp/sample_submission.csv'), pd.read_csv('data/comp_p1/sample_submission.csv')
print('sample submission unchanged:', a.equals(b))"

echo "== 2. news templates still all mapped?"
uv run python eda/05_validate_news.py | tail -15

echo "== 3. which reading of the metric matches Kaggle? (public window 209-221)"
uv run python eda/17_metric_variants.py 209 221 data/comp_p1/sample_submission.csv \
  submissions/teammate/primary_w208.csv=44.95 submissions/policy_v1_w208.csv=54.10 \
  submissions/lean_hedged_w208.csv=60.66 submissions/insured_hedged_w208.csv=48.55 \
  submissions/lean_split_w208.csv=67.74 \
  submissions/lean_split_d92_w208.csv=67.40 | tee reports/metric_check_p2.txt

echo "== 4. rebuild caches and backtests with weeks 1-221"
rm -f reports/policy_cache.pkl
uv run dsdg backtest lean_hedged > reports/bt_lean_hedged_p2.txt
uv run dsdg backtest insured_hedged > reports/bt_insured_hedged_p2.txt
uv run pytest -q

echo "== 4b. rank strategies and pairs under the matching metric reading (edit the 3 words to the reading eda/17 picked)"
for s in lean_split lean_split_d92 lean_hedged insured_hedged; do uv run dsdg backtest $s --step 4 > /dev/null; done
uv run python eda/27_pick_pair.py pooled pooled zero

echo "== 5. final pair from data through week 221"
uv run dsdg submit lean_hedged
uv run dsdg submit lean_split
uv run dsdg submit lean_split_d92
uv run dsdg submit insured_hedged
ls -la submissions/*_w221.csv
