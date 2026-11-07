"""Score candidates under joint news-story onset and duration scenarios.

Each active event is drawn once per scenario and applied to every supplier it
targets. The independent source planner draws per material; this model keeps a
Southern lane strike correlated across cobalt, copper, lithium, and gas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, draw_scenarios, shock_history  # noqa: E402
from dsdg_supply_chain.simulator import Truth, hindsight_plan  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402

from leaderboard_inverse import MATERIALS, SCORES, arrays, diversification, load_policy, score_policy  # noqa: E402
from lithium_share_scenarios import simulate  # noqa: E402


def event_masks(calendar: dict, count: int, rng) -> dict[str, np.ndarray]:
    output = {}
    for event in calendar["active"]:
        pmf = np.zeros(13)
        for week, probability in event["onset_weeks"].items():
            pmf[int(week)-209] = probability
        pmf /= pmf.sum()
        onset = rng.choice(13, size=count, p=pmf)
        real = rng.random(count) < event["conversion_probability"]
        survival = np.array([event["duration_survival"][str(d)] for d in range(1, 14)])
        u = rng.random(count)
        duration = (survival[None, :] > u[:, None]).sum(axis=1)
        week_index = np.arange(13)
        output[event["entity"]] = real[:, None] & (week_index[None, :] >= onset[:, None]) & (
            week_index[None, :] < onset[:, None]+duration[:, None])
    return output


def main() -> None:
    calendar = json.loads((ROOT / "reports" / "event_calendar_w208.json").read_text())
    info = load_data(ROOT / "data" / "snapshots" / "w208")
    weeks = np.arange(209, 222)
    features = build_features(info)
    forecast = supplier_forecast(info, range(209, 222), feats=features)
    state = supplier_state(info, features, forecast).merge(
        info.suppliers[["material", "country", "lane"]], on=["material", "country"])
    policies = {score: load_policy(ROOT / "submissions" / f"submission{score}.csv")
                for score in SCORES}
    policies["buffer"] = load_policy(
        ROOT / "submissions" / "candidates" / "correlated_lane_buffer_w208.csv")
    policies["naive"] = naive(info, range(209, 222))
    count = 500
    results = []
    for hazard in (0., .25, 1.):
        rng = np.random.default_rng(2067)
        events = event_masks(calendar, count, rng)
        params = LPParams(n_scenarios=count, cap_derate=.96, hazard_mult=hazard)
        history = shock_history(info) if hazard else None
        cost = {name: np.zeros((count, len(MATERIALS))) for name in policies}
        served = {name: np.zeros((count, len(MATERIALS))) for name in policies}
        demand_shock = np.zeros((count, len(MATERIALS)))
        hindsight = np.zeros((count, len(MATERIALS)))
        for mi, material in enumerate(MATERIALS):
            st = state[state.material == material].copy()
            target = {(row["material"], row["country"]): event for event in calendar["active"]
                      for row in event["affected"]}
            for country in st.country.unique():
                if (material, country) in target:
                    st.loc[st.country == country, "p"] = 0.
            sc = draw_scenarios(st, list(weeks), params, rng, history)
            for supplier, country in enumerate(sc["countries"]):
                event = target.get((material, country))
                if event is None:
                    continue
                mask = events[event["entity"]]
                sc["hit"][:, supplier, :] |= mask
                severity = .5 if event["type"] == "shortage" else .8
                sc["cap_ratio"][:, supplier, :] = np.where(
                    mask, np.minimum(sc["cap_ratio"][:, supplier, :], severity),
                    sc["cap_ratio"][:, supplier, :])
                if event["type"] == "strike":
                    lead = int(st[st.country == country].base_lead_weeks.iloc[0]) + 4
                    sc["lead"][:, supplier, :] = np.where(
                        mask, np.maximum(sc["lead"][:, supplier, :], lead),
                        sc["lead"][:, supplier, :])
            demand = info.demand.query("material == @material and 209 <= week <= 221").sort_values("week").demand_units.to_numpy(float)
            hit = sc["hit"].any(axis=1)
            shock = np.zeros_like(hit)
            for lag in range(5):
                shock[:, lag:] |= hit[:, :13-lag]
            demand_shock[:, mi] = (shock*demand[None, :]).sum(axis=1)
            for name, frame in policies.items():
                orders, shares = arrays(frame, material, sc["countries"], weeks)
                cost[name][:, mi], full_served = simulate(sc, orders, shares, demand)
                served[name][:, mi] = (full_served*shock).sum(axis=1)
            for scenario in range(count):
                price = sc["cost"][scenario] if sc["cost"].ndim == 3 else sc["cost"]
                truth = Truth(material, weeks, sc["countries"], demand,
                              sc["cap"]*sc["cap_ratio"][scenario],
                              sc["lead"][scenario].astype(int), price,
                              hit[scenario].astype(int), shock[scenario])
                hindsight[scenario, mi] = hindsight_plan(truth)[0]
        for financial in ("pooled", "mean", "mean_clip"):
            for fill in ("pooled", "mean"):
                for diversity in ("row", "weighted"):
                    values = {name: score_policy(cost[name], cost["naive"], hindsight,
                                                 served[name], served["naive"], demand_shock,
                                                 diversification(frame, diversity == "weighted"),
                                                 financial, fill)
                              for name, frame in policies.items() if name != "naive"}
                    delta = values["buffer"]-values["67.74"]
                    known = [values["67.74"]-values["60.66"],
                             values["66.98"]-values["67.74"],
                             values["67.26"]-values["67.74"]]
                    results.append(dict(hazard=hazard, financial=financial, fill=fill,
                                        diversity=diversity, mean=float(delta.mean()),
                                        positive_probability=float((delta > 0).mean()),
                                        p10=float(np.quantile(delta, .1)),
                                        known_deltas=[float(x.mean()) for x in known],
                                        by_event={key: dict(
                                            on=float(delta[mask.any(axis=1)].mean()),
                                            off=float(delta[~mask.any(axis=1)].mean()))
                                            for key, mask in events.items()}))
    output = ROOT / "reports" / "correlated_event_scenarios_w208.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        if row["financial"] == "pooled" and row["fill"] == "pooled":
            print(row["hazard"], row["diversity"], "mean", round(row["mean"], 2),
                  "p+", round(row["positive_probability"], 2),
                  "p10", round(row["p10"], 2),
                  "known", [round(x, 2) for x in row["known_deltas"]],
                  "by event", row["by_event"])
    print(output)


if __name__ == "__main__":
    main()
