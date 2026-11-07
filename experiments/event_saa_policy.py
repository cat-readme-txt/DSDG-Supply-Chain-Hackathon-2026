"""Private policy scenarios drawn from whole news events, not weekly marginals.

An event has a single conversion decision, onset, and duration. Its mask is
shared across every supplier it targets. Remaining supplier risk is sampled
from the existing chronological forecast so the marginal is approximately
preserved. The SAA LP then chooses orders under these correlated scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.policy import TYPE_CAP, TYPE_LEAD, supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, draw_scenarios, solve_window  # noqa: E402
from dsdg_supply_chain.submission import read_submission, write_submission  # noqa: E402
from experiments.event_calendar import calendar_for  # noqa: E402


def event_masks(calendar: dict, weeks: list[int], count: int, seed: int,
                event_scale: float = 1.) -> list[tuple[dict, np.ndarray]]:
    rng = np.random.default_rng(seed)
    T = len(weeks)
    output = []
    for event in calendar["active"]:
        if event["type"] == "tariff" or not event["onset_weeks"] or not event["affected"]:
            continue  # announced tariffs are handled directly by supplier_state
        pmf = np.array([event["onset_weeks"].get(str(w), 0.) for w in weeks])
        if pmf.sum() <= 0:
            continue
        pmf /= pmf.sum()
        onset = rng.choice(T, size=count, p=pmf)
        conversion = float(np.clip(event["conversion_probability"]*event_scale, 0, 1))
        real = rng.random(count) < conversion
        survival = np.array([event["duration_survival"].get(str(d), 0.) for d in range(1, T+1)])
        u = rng.random(count)
        duration = (survival[None, :] > u[:, None]).sum(axis=1)
        time = np.arange(T)
        mask = real[:, None] & (time[None, :] >= onset[:, None]) & (time[None, :] < onset[:, None]+duration[:, None])
        output.append((event, mask))
    return output


def event_plan(info, weeks: range, count: int = 80, event_scale: float = 1.,
               min_share: float = 0., seed: int = 221) -> pd.DataFrame:
    if weeks[0] <= info.last_week:
        raise ValueError("Plan weeks must be after the information cutoff")
    win = list(weeks)
    feats = build_features(info)
    sup = supplier_forecast(info, weeks, feats=feats)
    state = supplier_state(info, feats, sup).merge(
        info.suppliers[["material", "country", "lane"]], on=["material", "country"])
    calendar = calendar_for(info)
    masks = event_masks(calendar, win, count, seed, event_scale)
    rng = np.random.default_rng(seed + 177)
    params = LPParams(n_scenarios=count, min_share=min_share, hazard_mult=0.,
                      cap_derate=.96, cheap_premium=.025, cheap_share=.30, seed=seed)
    demand = info.demand.set_index(["material", "week"]).demand_units
    output = []
    for material in sorted(state.material.unique()):
        st = state[state.material == material].copy()
        material_masks = []
        for event, mask in masks:
            countries = [x["country"] for x in event["affected"] if x["material"] == material]
            if countries:
                material_masks.append((event, mask, countries))
        # Remove the part of each supplier marginal explained by explicit
        # event masks, leaving its overall hit probability near the forecast.
        for country in st.country.unique():
            relevant = [mask for _, mask, countries in material_masks if country in countries]
            if not relevant:
                continue
            union = np.logical_or.reduce(relevant)
            event_p = union.mean(axis=0)
            rows = st.country == country
            original = st.loc[rows].sort_values("week").p.to_numpy(float)
            residual = np.clip((original-event_p)/np.maximum(1-event_p, 1e-8), 0, 1)
            st.loc[st.loc[rows].sort_values("week").index, "p"] = residual
        sc = draw_scenarios(st, win, params, rng)
        for event, mask, countries in material_masks:
            typ = event["type"]
            for country in countries:
                s = sc["countries"].index(country)
                sc["hit"][:, s, :] |= mask
                sc["cap_ratio"][:, s, :] = np.where(mask,
                    np.minimum(sc["cap_ratio"][:, s, :], TYPE_CAP.get(typ, .5)),
                    sc["cap_ratio"][:, s, :])
                extra = TYPE_LEAD.get(typ, 0)
                if extra:
                    base_lead = int(st[st.country == country].base_lead_weeks.iloc[0])
                    sc["lead"][:, s, :] = np.where(mask,
                        np.maximum(sc["lead"][:, s, :], base_lead+extra),
                        sc["lead"][:, s, :])
        d = np.array([float(demand.loc[(material, w)]) for w in win])
        x = solve_window(d, sc, params)
        q = x.sum(axis=0)
        rows = pd.DataFrame(dict(week=win, material=material, order_units=q))
        for col in SHARE_COLS:
            rows[col] = 0.
        for s, country in enumerate(sc["countries"]):
            rows[f"share_{country}"] = np.divide(x[s], q,
                out=np.full(len(win), 1/len(sc["countries"])), where=q>1e-6)
        output.append(rows)
    return pd.concat(output, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=80)
    p.add_argument("--event-scale", type=float, default=1.)
    p.add_argument("--min-share", type=float, default=0.)
    p.add_argument("--output", type=Path, default=ROOT/"reports"/"generated_candidates"/"event_saa_w221.csv")
    args = p.parse_args()
    info = load_data(ROOT/"data"/"Phase 2 data")
    base = read_submission(ROOT/"submissions"/"submission91.36.csv")
    plan = event_plan(info, range(222,235), args.count, args.event_scale, args.min_share)
    full = base.set_index(["week", "material"])
    full["order_units"] = full.order_units.astype(float)
    plan = plan.set_index(["week", "material"])
    full.loc[plan.index, ["order_units", *SHARE_COLS]] = plan[["order_units", *SHARE_COLS]]
    full = full.reset_index().sort_values(["week", "material"])
    write_submission(full, args.output)
    from pipeline import CompetitionData
    CompetitionData(ROOT/"data"/"Phase 2 data").validate_file(args.output)
    print(json.dumps(dict(output=str(args.output), origin=info.last_week,
                          events=[x[0]["entity"] for x in event_masks(calendar_for(info),
                                  list(range(222,235)), args.count, 221, args.event_scale)]), indent=2))


if __name__ == "__main__":
    main()
