"""Score candidate files on simulated futures of the private window (weeks 222-234), built from the
state at week 221 (eda/29_state_report.py): the announced KES tariff on aluminum and semiconductors
(certain), the futures-confirmed VAL cobalt and KES rare-earths bans (onset and duration drawn from
past bans), and the MERROW closure story either coming true or not.

Each scenario builds a synthetic truth per material (capacity with normal noise, lead, landed cost,
disrupted label, benchmark price), then scores every file with the simulator, the hindsight LP and
the metric (F and fill pooled, diversification row-mean and order-weighted, calm fill = 0).
usage: uv run python eda/30_scenario_pair.py file1.csv file2.csv ...
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.simulator import Truth, hindsight_plan, simulate
from dsdg_supply_chain.scoring import policy_arrays
from dsdg_supply_chain.submission import read_submission

d = load_data()
O, W = d.last_week, list(range(222, 235))
T = len(W)
naive = read_submission(os.path.join(os.environ.get("DSDG_DATA_DIR", "data/comp"), "sample_submission.csv"))
files = {f.split("/")[-1].replace("_w221.csv", ""): read_submission(f) for f in sys.argv[1:]}
sup = d.suppliers
key = ["week", "material", "country"]
now = (d.supply_history[d.supply_history.week == O].merge(d.tariffs[key + ["tariff_rate"]], on=key)
       .merge(d.prices[key + ["spot_price"]], on=key).merge(sup, on=["material", "country"]))
now["freight"] = now.landed_cost - now.spot_price * (1 + now.tariff_rate)
bench = d.outcomes[d.outcomes.week == O].set_index("material").benchmark_price
BAN_UPLIFT = {"cobalt": 0.30, "rare_earths": 0.29}
MERROW_FREIGHT_MULT = 3.0


def truth(mat, sc, rng):
    rows = now[now.material == mat].sort_values("normal_capacity_units", ascending=False)
    cs = rows.country.tolist()
    S = len(cs)
    cap = rows.normal_capacity_units.to_numpy()[:, None] * rng.uniform(0.95, 1.0, (S, T))
    lead = np.repeat(rows.base_lead_weeks.to_numpy()[:, None], T, 1)
    spot = np.repeat(rows.spot_price.to_numpy()[:, None], T, 1)
    tar = np.repeat(rows.tariff_rate.to_numpy()[:, None], T, 1)
    fr = np.repeat(rows.freight.to_numpy()[:, None], T, 1)
    hit = np.zeros((S, T), bool)
    up = np.ones(T)
    for (bm, bc), (on, dur) in sc["bans"].items():
        if bm == mat:
            act = np.array([on <= w < on + dur for w in W])
            i = cs.index(bc); cap[i, act] *= 0.05; hit[i, act] = True; up[act] *= 1 + BAN_UPLIFT[mat]
    if mat in ("aluminum", "semiconductors"):
        i = cs.index("KES"); act = np.array([w >= 224 for w in W]); tar[i, act] = 0.35; hit[i, act] = True
    if sc["closure"]:
        on, dur = sc["closure"]
        act = np.array([on <= w < on + dur for w in W])
        for i, c in enumerate(cs):
            if rows.lane.iloc[i] == "MERROW":
                cap[i, act] *= 0.75; lead[i, act] += 3; fr[i, act] *= MERROW_FREIGHT_MULT; hit[i, act] = True
    cost = spot * up[None] * (1 + tar) + fr
    dis = hit.any(0).astype(int)
    shock = np.array([dis[max(0, t - 4):t + 1].any() for t in range(T)])
    dem = d.demand[(d.demand.material == mat) & d.demand.week.isin(W)].sort_values("week").demand_units.to_numpy(float)
    return Truth(mat, np.array(W), cs, dem, cap, lead, cost, dis, shock), float(bench[mat]) * up


def score(sub, tr_all, price_all):
    p_rows, cost = [], dict(c=0.0, n=0.0, h=0.0, sv=0.0, svn=0.0, ds=0.0)
    for mat, (tr, hind) in tr_all.items():
        s = sub[(sub.material == mat) & sub.week.isin(W)].sort_values("week")
        nv = naive[(naive.material == mat) & naive.week.isin(W)].sort_values("week")
        mine, base = simulate(tr, *policy_arrays(sub, tr)), simulate(tr, *policy_arrays(naive, tr))
        cost["c"] += mine.total_cost; cost["n"] += base.total_cost; cost["h"] += hind
        sw = tr.shock_week
        cost["sv"] += mine.served[sw].sum(); cost["svn"] += base.served[sw].sum(); cost["ds"] += tr.demand[sw].sum()
        px = price_all[mat]
        p_rows.append(pd.DataFrame(dict(y=tr.disrupted, p=s.disruption_prob.to_numpy(), pn=nv.disruption_prob.to_numpy(),
                                        px=px, f=s.price_forecast.to_numpy(), fn=nv.price_forecast.to_numpy())))
    P = pd.concat(p_rows)
    bs = 1 - ((P.p - P.y) ** 2).mean() / ((P.pn - P.y) ** 2).mean()
    ps = 1 - (abs(P.f - P.px) / P.px).mean() / (abs(P.fn - P.px) / P.px).mean()
    Pp = np.clip(0.6 * bs + 0.4 * ps, 0, 1)
    F = np.clip((cost["n"] - cost["c"]) / (cost["n"] - cost["h"]), 0, 1)
    fr, frn = cost["sv"] / cost["ds"], cost["svn"] / cost["ds"]
    fill = (fr - frn) / (1 - frn) if frn < 1 - 1e-9 else 0.0
    sh = sub[sub.week.isin(W)]
    dv = np.clip((1 - (sh[[c for c in sh if c.startswith("share_")]].to_numpy() ** 2).sum(1)) / (2 / 3), 0, 1)
    w = sh.order_units.to_numpy()
    out = {}
    for name, div in [("row", dv.mean()), ("wtd", (dv * w).sum() / w.sum())]:
        R = np.clip(0.7 * fill + 0.3 * div, 0, 1)
        out[name] = 100 * (0.3 * Pp + 0.4 * F + 0.3 * R)
    return out | dict(P=Pp, F=F, fill=fill)


rng = np.random.default_rng(0)
rows = []
# Phase 2 stage gate (DECISIONS P2-2): no ban or closure has reached its last pre-onset stage at 221,
# so onsets start at 223 at the earliest (222 kept at a small weight for the 1-in-9 skip rate)
closures = [None] + [(on, dur) for on in (223, 224, 225) for dur in (3, 5, 8)]
for n in range(48):
    sc = dict(closure=closures[n % len(closures)],
              bans={("cobalt", "VAL"): (int(rng.choice([222, 223, 224], p=[0.1, 0.45, 0.45])),
                                        int(rng.choice([5, 6, 8, 11]))),
                    ("rare_earths", "KES"): (int(rng.choice([223, 224, 225, 226], p=[0.15, 0.35, 0.35, 0.15])),
                                             int(rng.choice([5, 6, 8, 11])))})
    tr_all, price_all = {}, {}
    for mat in sorted(sup.material.unique()):
        tr, px = truth(mat, sc, rng)
        tr_all[mat] = (tr, hindsight_plan(tr)[0]); price_all[mat] = px
    for name, sub in files.items():
        rows.append(dict(file=name, closure=sc["closure"] is not None, **score(sub, tr_all, price_all)))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.groupby(["closure", "file"])[["wtd", "row", "P", "F", "fill"]].mean().round(2).to_string())
p_close = float(sys.argv[0] and 0.4)
print(f"\nexpected score (P(closure)={p_close}), order-weighted diversification:")
m = R.groupby(["closure", "file"]).wtd.mean().unstack(0)
print((m[False] * (1 - p_close) + m[True] * p_close).sort_values(ascending=False).round(2).to_string())
print("\nbest pairs, expected best-of-two (per scenario max), order-weighted:")
piv = R.assign(sid=R.index // len(files)).pivot(index="sid", columns="file", values="wtd")
cl = R.groupby(R.index // len(files)).closure.first()
wts = np.where(cl, p_close / cl.sum(), (1 - p_close) / (~cl).sum())
pairs = sorted(((float((np.maximum(piv[a], piv[b]) * wts).sum()), a, b) for a, b in itertools.combinations(piv, 2)), reverse=True)
for v, a, b in pairs[:6]:
    print(f"  {v:6.2f}  {a} + {b}")
