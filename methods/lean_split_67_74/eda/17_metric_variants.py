"""Which reading of the Evaluation tab does Kaggle actually use?

The rules leave several choices open (E2 in DECISIONS.md). Each one changes which materials and
weeks matter, so the policy should be tuned to the right one. This script scores submissions under
every combination of these choices:

  F_agg    pooled dollars over materials | mean of per-material F | mean of clipped per-material F
  fill_agg pooled units over materials   | mean of per-material fill skill
  calm     fill skill when naive fills 100% (or no shock weeks): 0 | 1 | material skipped
  div      mean over rows | weighted by order units
  avgcost  mean over suppliers x weeks | mean of the benchmark-equivalent (spot avg) -- landed only here
  init     sum of first 5 weeks of demand | 5 x mean window demand

Usage (after Phase 2, with weeks 209-221 released):
  uv run python eda/17_metric_variants.py 209 221 sample.csv submissions/teammate/primary_w208.csv=44.95 \
      submissions/policy_v1_w208.csv=54.10
Without arguments it runs a demo on a released backtest window so the spread of variants is visible.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.scoring import policy_arrays
from dsdg_supply_chain.simulator import HOLDING_RATE, SALVAGE_RATE, UNMET_MULT, Truth, truth_for
from dsdg_supply_chain.submission import read_submission
from dsdg_supply_chain.strategies import naive as naive_strategy, teammate_v1, policy_v1, lean_lp, insured
from scipy.optimize import linprog


def sim(tr: Truth, orders, shares, A, init):
    S, T = tr.capacity.shape
    filled = np.minimum(orders[None, :] * shares, tr.capacity)
    arr = np.zeros(T + tr.lead.max() + 1)
    for s in range(S):
        for t in range(T):
            arr[t + tr.lead[s, t]] += filled[s, t]
    stock, hold, unmet, served = init, 0.0, 0.0, np.zeros(T)
    for t in range(T):
        stock += arr[t]
        served[t] = min(stock, tr.demand[t])
        stock -= served[t]
        hold += HOLDING_RATE * A * stock
        unmet += UNMET_MULT * A * (tr.demand[t] - served[t])
    cost = (filled * tr.cost).sum() + hold + unmet - SALVAGE_RATE * A * (stock + arr[T:].sum())
    return float(cost), served


def hindsight(tr: Truth, A, init):
    S, T = tr.capacity.shape
    nx = S * T
    iu, ii = nx, nx + T
    n = nx + 2 * T
    c = np.zeros(n)
    for s in range(S):
        for t in range(T):
            c[s * T + t] = tr.cost[s, t] - (SALVAGE_RATE * A if t + tr.lead[s, t] >= T else 0.0)
    c[iu:iu + T] = UNMET_MULT * A
    c[ii:ii + T] = HOLDING_RATE * A
    c[ii + T - 1] -= SALVAGE_RATE * A
    Aeq, beq = np.zeros((T, n)), -tr.demand.copy()
    for t in range(T):
        Aeq[t, ii + t] = 1
        if t:
            Aeq[t, ii + t - 1] = -1
        Aeq[t, iu + t] = -1
    beq[0] += init
    for s in range(S):
        for t in range(T):
            if (a := t + tr.lead[s, t]) < T:
                Aeq[a, s * T + t] -= 1
    bounds = ([(0, tr.capacity[s, t]) for s in range(S) for t in range(T)]
              + [(0, d) for d in tr.demand] + [(0, None)] * T)
    return float(linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs").fun)


def per_material(sub, naive, data, weeks):
    """Raw per-material ingredients under each (avgcost, init) choice."""
    rows = []
    for mat in sorted(sub.material.unique()):
        tr = truth_for(data, mat, weeks)
        mine, base = policy_arrays(sub, tr), policy_arrays(naive, tr)
        for avgcost, init in itertools.product(["supweek", "spot"], ["sum5", "5mean"]):
            if avgcost == "supweek":
                A = tr.cost.mean()
            else:
                px = data.outcomes[(data.outcomes.material == mat) & data.outcomes.week.isin(weeks)]
                A = px.benchmark_price.mean()
            I0 = tr.demand[:5].sum() if init == "sum5" else 5 * tr.demand.mean()
            c, sv = sim(tr, *mine, A, I0)
            cn, svn = sim(tr, *base, A, I0)
            h = hindsight(tr, A, I0)
            sw = tr.shock_week
            rows.append(dict(material=mat, avgcost=avgcost, init=init, cost=c, naive=cn, hind=h,
                             served=sv[sw].sum(), served_n=svn[sw].sum(), dem_s=tr.demand[sw].sum()))
    return pd.DataFrame(rows)


def p_part(sub, naive, data, weeks):
    key = ["week", "material"]
    out = data.outcomes[data.outcomes.week.isin(weeks)][key + ["disrupted", "benchmark_price"]]
    m = out.merge(sub[sub.week.isin(weeks)], on=key).merge(
        naive[key + ["disruption_prob", "price_forecast"]], on=key, suffixes=("", "_n"))
    res = {}
    for agg in ["pooled", "per_mat"]:
        def skill(g):
            y, px = g.disrupted, g.benchmark_price
            bs, bsn = ((g.disruption_prob - y) ** 2).mean(), ((g.disruption_prob_n - y) ** 2).mean()
            mp, mpn = (abs(g.price_forecast - px) / px).mean(), (abs(g.price_forecast_n - px) / px).mean()
            return pd.Series({"b": 1 - bs / bsn if bsn > 0 else 0.0, "p": 1 - mp / mpn if mpn > 0 else 0.0})
        s = skill(m) if agg == "pooled" else m.groupby("material").apply(skill).mean()
        res[agg] = (s.b, s.p)
    return res


def div_part(sub, weeks):
    s = sub[sub.week.isin(weeks)]
    sh = s[SHARE_COLS].to_numpy(float)
    d = np.clip((1 - (sh ** 2).sum(1)) / (2 / 3), 0, 1)
    w = s.order_units.to_numpy(float)
    return {"row": float(d.mean()), "weighted": float((d * w).sum() / w.sum()) if w.sum() else 0.0}


def all_variants(sub, naive, data, weeks) -> pd.DataFrame:
    pm = per_material(sub, naive, data, weeks)
    pp = p_part(sub, naive, data, weeks)
    dv = div_part(sub, weeks)
    rows = []
    for (avgcost, init), g in pm.groupby(["avgcost", "init"]):
        for F_agg, fill_agg, calm, div, p_agg in itertools.product(
                ["pooled", "mean", "mean_clip"], ["pooled", "mean"], ["zero", "one", "skip"],
                ["row", "weighted"], ["pooled", "per_mat"]):
            if F_agg == "pooled":
                F = (g.naive.sum() - g.cost.sum()) / (g.naive.sum() - g.hind.sum())
            else:
                f = (g.naive - g.cost) / (g.naive - g.hind)
                F = (f.clip(0, 1) if F_agg == "mean_clip" else f).mean()
            def fs(sv, svn, ds):
                if ds <= 0 or svn >= ds - 1e-9:
                    return {"zero": 0.0, "one": 1.0 if sv >= ds - 1e-9 else 0.0, "skip": np.nan}[calm]
                return (sv / ds - svn / ds) / (1 - svn / ds)
            if fill_agg == "pooled":
                fill = fs(g.served.sum(), g.served_n.sum(), g.dem_s.sum())
                fill = 0.0 if np.isnan(fill) else fill
            else:
                v = [fs(a, b, c) for a, b, c in zip(g.served, g.served_n, g.dem_s)]
                fill = 0.0 if np.all(np.isnan(v)) else float(np.nanmean(v))
            b, p = pp[p_agg]
            P = np.clip(0.6 * b + 0.4 * p, 0, 1)
            R = np.clip(0.7 * fill + 0.3 * dv[div], 0, 1)
            score = 100 * (0.3 * P + 0.4 * np.clip(F, 0, 1) + 0.3 * R)
            rows.append(dict(avgcost=avgcost, init=init, F_agg=F_agg, fill_agg=fill_agg, calm=calm, div=div,
                             p_agg=p_agg, score=score, P=P, F=F, fill=fill, divv=dv[div]))
    return pd.DataFrame(rows)


def main():
    data = load_data()
    if len(sys.argv) > 1:
        a, b = int(sys.argv[1]), int(sys.argv[2])
        weeks = range(a, b + 1)
        naive = read_submission(sys.argv[3])
        subs = {}
        for arg in sys.argv[4:]:
            path, _, ks = arg.partition("=")
            subs[path] = (read_submission(path), float(ks) if ks else np.nan)
    else:  # demo: origin 104, public window 105-117 (KES sanctions)
        o = 104
        weeks = range(o + 1, o + 14)
        allw = range(o + 1, o + 27)
        info = data.until(o)
        naive = naive_strategy(info, allw)
        subs = {n: (f(info, allw), np.nan) for n, f in [("teammate_v1", teammate_v1), ("policy_v1", policy_v1)]}
    keys = ["avgcost", "init", "F_agg", "fill_agg", "calm", "div", "p_agg"]
    tabs = []
    for name, (sub, ks) in subs.items():
        v = all_variants(sub, naive, data, weeks).assign(sub=name, kaggle=ks)
        v["err"] = v.score - ks
        tabs.append(v)
    t = pd.concat(tabs)
    wide = t.pivot_table(index=keys, columns="sub", values="score")
    if t.kaggle.notna().any():
        err = t.groupby(keys).err.apply(lambda e: np.sqrt((e ** 2).mean())).rename("rmse")
        wide = wide.join(err).sort_values("rmse")
        print("Kaggle:", {n: k for n, (_, k) in subs.items()})
    pd.set_option("display.width", 250, "display.max_rows", 60)
    print(wide.round(2).head(40).to_string())
    print("\nspread of scores across variants:\n", t.groupby("sub").score.describe().round(2))
    print("\ncomponent detail (first sub, pooled/zero/row baseline):")
    print(t[(t.F_agg == "pooled") & (t.fill_agg == "pooled") & (t.calm == "zero") & (t["div"] == "row")
            & (t.p_agg == "pooled")].round(3).to_string())


if __name__ == "__main__":
    main()
