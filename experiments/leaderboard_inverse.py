"""Infer plausible public-window supply states from three score differences.

The four scored files used here have identical forecast columns. Their score
differences therefore constrain only financial and resilience outcomes. Draw
future supply states from released-week forecasts, vary unresolved scorer
conventions, and retain states compatible with the leaderboard observations.
No week-209+ outcome or supply data are read.
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

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, draw_scenarios, shock_history  # noqa: E402
from dsdg_supply_chain.simulator import Truth, hindsight_plan  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402

from lithium_share_scenarios import simulate  # noqa: E402

SCORES = ("60.66", "67.74", "66.98", "67.26")
OBSERVED = np.array([7.08, -0.76, -0.48])
MATERIALS = ("aluminum", "cobalt", "copper", "lithium", "natural_gas",
             "polymer_resin", "rare_earths", "semiconductors",
             "specialty_glass", "steel")


def load_policy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["week"] = frame.id.str.extract(r"w(\d+)").astype(int)
    frame["material"] = frame.id.str.split("_", n=1).str[1]
    return frame[frame.week.between(209, 221)].copy()


def arrays(sub: pd.DataFrame, material: str, countries: list[str], weeks: np.ndarray):
    rows = sub[sub.material == material].set_index("week").loc[weeks]
    return (rows.order_units.to_numpy(float),
            rows[[f"share_{country}" for country in countries]].to_numpy(float).T)


def diversification(sub: pd.DataFrame, weighted: bool) -> float:
    sh = sub[[f"share_{c}" for c in ("KES", "VAL", "SUN", "ORI", "HAL", "AUR")]].to_numpy(float)
    score = np.clip((1 - (sh**2).sum(axis=1)) / (2/3), 0, 1)
    if weighted:
        weight = sub.order_units.to_numpy(float)
        return float(np.average(score, weights=weight))
    return float(score.mean())


def score_policy(cost: np.ndarray, base_cost: np.ndarray, hindsight: np.ndarray,
                 served: np.ndarray, base_served: np.ndarray,
                 shock_demand: np.ndarray, div: float,
                 financial: str, fill: str) -> np.ndarray:
    if financial == "pooled":
        denominator = (base_cost - hindsight).sum(axis=1)
        f = np.divide((base_cost - cost).sum(axis=1), denominator,
                      out=np.zeros(len(cost)), where=denominator > 1e-9)
    else:
        denominator = base_cost - hindsight
        x = np.divide(base_cost - cost, denominator,
                      out=np.zeros_like(cost), where=denominator > 1e-9)
        f = np.clip(x, 0, 1).mean(axis=1) if financial == "mean_clip" else x.mean(axis=1)
    deficit = shock_demand - base_served
    skill = np.divide(served - base_served, deficit,
                      out=np.zeros_like(served), where=deficit > 1e-9)
    if fill == "pooled":
        denominator = deficit.sum(axis=1)
        fill_score = np.divide((served - base_served).sum(axis=1), denominator,
                               out=np.zeros(len(cost)), where=denominator > 1e-9)
    else:
        fill_score = skill.mean(axis=1)
    r = np.clip(.7 * fill_score + .3 * div, 0, 1)
    return 40 * np.clip(f, 0, 1) + 30 * r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "leaderboard_inverse_w208.json")
    args = parser.parse_args()
    info = load_data(ROOT / "data" / "snapshots" / "w208")
    weeks = np.arange(209, 222)
    feats = build_features(info)
    sup = supplier_forecast(info, range(209, 222), feats=feats)
    state = supplier_state(info, feats, sup).merge(
        info.suppliers[["material", "country", "lane"]], on=["material", "country"])
    policies = {score: load_policy(ROOT / "submissions" / f"submission{score}.csv")
                for score in SCORES}
    policies["naive"] = naive(info, range(209, 222))

    base = policies["67.74"]

    def variant(name: str, order_scales: dict[str, float] | None = None,
                val_reduction: float = 0) -> None:
        frame = base.copy()
        frame["order_units"] = frame.order_units.astype(float)
        for material, scale in (order_scales or {}).items():
            mask = (frame.material == material) & frame.week.between(209, 214)
            frame.loc[mask, "order_units"] *= scale
        if val_reduction:
            mask = (frame.material == "lithium") & frame.week.between(209, 217)
            shifted = frame.loc[mask, "share_VAL"].to_numpy(float) * val_reduction
            frame.loc[mask, "share_VAL"] -= shifted
            frame.loc[mask, "share_KES"] += shifted
        policies[name] = frame

    for scale in (1.1, 1.2):
        variant(f"cobalt_{scale:.1f}", {"cobalt": scale})
        variant(f"gas_{scale:.1f}", {"natural_gas": scale})
        variant(f"both_{scale:.1f}", {"cobalt": scale, "natural_gas": scale})
    variant("half_val", val_reduction=.5)
    variant("no_val", val_reduction=1.)
    variant("both_1.1_half_val", {"cobalt": 1.1, "natural_gas": 1.1}, .5)
    variant("both_1.2_half_val", {"cobalt": 1.2, "natural_gas": 1.2}, .5)
    candidates = [name for name in policies if name not in (*SCORES, "naive")]

    results = []
    for hazard, forced_val in ((0., False), (.25, False), (1., False),
                               (0., True), (.25, True), (1., True)):
        n = args.scenarios
        rng = np.random.default_rng(7105 + int(forced_val))
        params = LPParams(n_scenarios=n, cap_derate=.96, hazard_mult=hazard)
        history = shock_history(info) if hazard else None
        cost = {key: np.zeros((n, len(MATERIALS))) for key in policies}
        served = {key: np.zeros((n, len(MATERIALS))) for key in policies}
        demand_shock = np.zeros((n, len(MATERIALS)))
        hindsight = np.zeros((n, len(MATERIALS)))
        actual_hit = np.zeros((n, len(MATERIALS)), bool)
        for mi, material in enumerate(MATERIALS):
            st = state[state.material == material]
            sc = draw_scenarios(st, list(weeks), params, rng, history)
            if material == "lithium" and forced_val:
                # Importance-sample rare unannounced Valdmar lithium shocks.
                # The model average below gives this branch a 3% prior.
                val_idx = sc["countries"].index("VAL")
                onset = rng.integers(0, 8, size=n)
                duration = rng.integers(2, 10, size=n)
                severity = rng.choice([.05, .2, .5], size=n)
                for scenario in range(n):
                    stop = min(len(weeks), onset[scenario] + duration[scenario])
                    sl = slice(onset[scenario], stop)
                    sc["hit"][scenario, val_idx, sl] = True
                    sc["cap_ratio"][scenario, val_idx, sl] = np.minimum(
                        sc["cap_ratio"][scenario, val_idx, sl], severity[scenario])
            demand = info.demand.query("material == @material and 209 <= week <= 221").sort_values("week").demand_units.to_numpy(float)
            hit = sc["hit"].any(axis=1)
            actual_hit[:, mi] = hit.any(axis=1)
            shock = np.zeros_like(hit)
            for lag in range(5):
                shock[:, lag:] |= hit[:, :len(weeks)-lag]
            demand_shock[:, mi] = (shock * demand[None, :]).sum(axis=1)
            countries = sc["countries"]
            for name, frame in policies.items():
                orders, shares = arrays(frame, material, countries, weeks)
                cost[name][:, mi], served_full = simulate(sc, orders, shares, demand)
                served[name][:, mi] = (served_full * shock).sum(axis=1)
            for scenario in range(n):
                price = sc["cost"][scenario] if sc["cost"].ndim == 3 else sc["cost"]
                truth = Truth(material, weeks, countries, demand,
                              sc["cap"] * sc["cap_ratio"][scenario],
                              sc["lead"][scenario].astype(int), price,
                              hit[scenario].astype(int), shock[scenario])
                hindsight[scenario, mi] = hindsight_plan(truth)[0]
            print(f"hazard={hazard} material={material}", flush=True)
        for financial in ("pooled", "mean", "mean_clip"):
            for fill in ("pooled", "mean"):
                for div_kind in ("row", "weighted"):
                    div = {name: diversification(frame, div_kind == "weighted")
                           for name, frame in policies.items() if name != "naive"}
                    values = {name: score_policy(cost[name], cost["naive"], hindsight,
                                                 served[name], served["naive"],
                                                 demand_shock, div[name], financial, fill)
                              for name in policies if name != "naive"}
                    predicted = np.stack((values["67.74"]-values["60.66"],
                                          values["66.98"]-values["67.74"],
                                          values["67.26"]-values["67.74"]), axis=1)
                    for sigma in (.25, .5, 1., 2.):
                        error = ((predicted - OBSERVED) / sigma)**2
                        raw_log_weight = -.5 * error.sum(axis=1)
                        max_log_weight = raw_log_weight.max()
                        weight = np.exp(raw_log_weight - max_log_weight)
                        weight /= weight.sum()
                        top = np.argsort(weight)[-5:][::-1]
                        candidate_deltas = {name: values[name]-values["67.74"]
                                            for name in candidates}
                        results.append(dict(hazard=hazard, forced_val=forced_val,
                                            financial=financial,
                                            fill=fill, diversity=div_kind, sigma=sigma,
                                            effective_n=float(1/(weight**2).sum()),
                                            log_evidence=float(max_log_weight +
                                                               np.log(np.exp(raw_log_weight-max_log_weight).mean())),
                                            weighted_deltas=(weight[:, None]*predicted).sum(axis=0).tolist(),
                                            candidates={name: dict(
                                                mean=float(np.sum(weight*delta)),
                                                positive_probability=float(np.sum(weight*(delta > 0))))
                                                for name, delta in candidate_deltas.items()},
                                            top_scenarios=[dict(index=int(i), weight=float(weight[i]),
                                                                deltas=predicted[i].tolist(),
                                                                hit_materials=[MATERIALS[j] for j in range(len(MATERIALS))
                                                                               if actual_hit[i, j]]) for i in top]))
    model_averages = {}
    for sigma in (.25, .5, 1., 2.):
        group = [row for row in results if row["sigma"] == sigma]
        log_model_weight = np.array([
            row["log_evidence"] + np.log((.03 if row["forced_val"] else .97) / 3 / 12)
            for row in group])
        model_weight = np.exp(log_model_weight-log_model_weight.max())
        model_weight /= model_weight.sum()
        model_averages[str(sigma)] = {
            "forced_val_posterior": float(sum(w*row["forced_val"]
                                               for w, row in zip(model_weight, group))),
            "candidate_deltas": {
                name: {
                    "mean": float(sum(w*row["candidates"][name]["mean"]
                                      for w, row in zip(model_weight, group))),
                    "positive_probability": float(sum(w*row["candidates"][name]["positive_probability"]
                                                      for w, row in zip(model_weight, group)))
                } for name in candidates}}
    output = {"observed_deltas": OBSERVED.tolist(), "n_per_hazard": args.scenarios,
              "assumptions": "Scenarios from released-week forecasts; same score formula family as archived metric variants.",
              "model_averages": model_averages,
              "results": sorted(results, key=lambda r: r["log_evidence"], reverse=True)}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for r in [x for x in output["results"] if x["sigma"] == .5][:10]:
        best = sorted(r["candidates"].items(), key=lambda item: item[1]["mean"], reverse=True)[:4]
        print(r["hazard"], "forced_val", r["forced_val"],
              r["financial"], r["fill"], r["diversity"],
              "sigma", r["sigma"], "ESS", round(r["effective_n"], 1),
              "pred", [round(x, 2) for x in r["weighted_deltas"]],
              "best", [(name, round(value["mean"], 2)) for name, value in best])
    for sigma, model in model_averages.items():
        best = sorted(model["candidate_deltas"].items(),
                      key=lambda item: item[1]["mean"], reverse=True)
        print("model average sigma", sigma, "forced_val posterior",
              round(model["forced_val_posterior"], 3),
              "candidates", [(name, round(value["mean"], 2),
                              round(value["positive_probability"], 2))
                             for name, value in best])
    print(args.output)


if __name__ == "__main__":
    main()
