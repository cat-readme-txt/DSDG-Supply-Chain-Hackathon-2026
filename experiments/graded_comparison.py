"""Compare graded CSV fields and replay their policies on released supply windows."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

from pipeline import CompetitionData
from experiments.scenario_planner import actual_scenario


def read_submission(path: Path, data: CompetitionData) -> dict[str, dict[str, str]]:
    data.validate_file(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["id"]: row for row in csv.DictReader(stream)}


def simulate(data: CompetitionData, rows: dict[str, dict[str, str]], start: int,
             origin: int) -> dict[str, float]:
    totals = {key: 0.0 for key in ("cost", "demand", "shortage", "shock_demand", "shock_met", "diversity")}
    for material in data.materials:
        scenario = actual_scenario(data, material, origin+1)
        demand = [data.demand[start+offset, material] for offset in range(13)]
        inventory = float(sum(demand[:5]))
        arrivals = [0.0]*13
        transit = cost = 0.0
        countries = tuple(data.suppliers[material])
        for offset in range(13):
            row = rows[f"w{start+offset}_{material}"]
            order = float(row["order_units"])
            weights = [float(row[f"share_{country}"]) for country in countries]
            totals["diversity"] += 1.5*(1-sum(weight*weight for weight in weights))/13/len(data.materials)
            for index, weight in enumerate(weights):
                filled = min(order*weight, scenario.capacity[offset][index])
                cost += filled*scenario.cost[offset][index]
                arrival = offset+scenario.lead[offset][index]
                if arrival < 13:
                    arrivals[arrival] += filled
                else:
                    transit += filled
            inventory += arrivals[offset]
            met = min(inventory, demand[offset])
            inventory -= met
            missed = demand[offset]-met
            totals["demand"] += demand[offset]
            totals["shortage"] += missed
            cost += 5*scenario.avg_cost[offset]*missed + .02*scenario.avg_cost[offset]*inventory
            if any(data.outcomes[origin+1+shock, material][1]
                   for shock in range(max(0, offset-4), offset+1)):
                totals["shock_demand"] += demand[offset]
                totals["shock_met"] += met
        totals["cost"] += cost-.7*statistics.mean(scenario.avg_cost)*(inventory+transit)
    return totals


def summary(folds: list[dict[str, float]]) -> dict[str, float]:
    shock_demand = sum(row["shock_demand"] for row in folds)
    return {
        "scenario_count": len(folds),
        "mean_cost": statistics.mean(row["cost"] for row in folds),
        "fill": 1-sum(row["shortage"] for row in folds)/sum(row["demand"] for row in folds),
        "shock_fill": sum(row["shock_met"] for row in folds)/shock_demand if shock_demand else 1.0,
        "diversity": statistics.mean(row["diversity"] for row in folds),
    }


def main() -> None:
    data = CompetitionData(Path("data"))
    paths = sorted(Path("submissions").glob("submission[0-9]*.csv"))
    paths.append(Path("submissions/candidates/private_risk_hedge_w208.csv"))
    origins = data.complete_fold_origins()
    shocks = {origin: sum(data.outcomes[week, material][1]
                          for week in range(origin+1, origin+14) for material in data.materials)
              for origin in origins}
    results = {}
    for path in paths:
        rows = read_submission(path, data)
        policy = {}
        for start in (209, 222):
            folds = {origin: simulate(data, rows, start, origin) for origin in origins}
            policy[str(start)] = {
                "all": summary(list(folds.values())),
                "quiet_fewer_than_15_disrupted_material_weeks": summary([folds[o] for o in origins if shocks[o] < 15]),
                "shock_at_least_15_disrupted_material_weeks": summary([folds[o] for o in origins if shocks[o] >= 15]),
            }
        results[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "order_units": sum(float(row["order_units"]) for row in rows.values()),
            "zero_order_rows": sum(float(row["order_units"]) == 0 for row in rows.values()),
            "historical_supply_replay": policy,
        }
    incumbent = read_submission(Path("submissions/submission60.66.csv"), data)
    hedge = read_submission(Path("submissions/candidates/private_risk_hedge_w208.csv"), data)
    public_equal = all(incumbent[key] == hedge[key] for key in incumbent if int(key.split("_")[0][1:]) <= 221)
    report = {
        "historical_end_week": data.historical_end,
        "scored_file_scores_are_user_reported": True,
        "private_hedge_public_rows_identical_to_60_66": public_equal,
        "results": results,
        "limitations": [
            "Historical supply replay is a stress test, not the official simulator or a forecast of hidden supply.",
            "Quiet/shock grouping uses released outcomes only; it does not reveal future labels.",
            "No official private score exists for the candidate files.",
        ],
    }
    output = Path("reports/graded_comparison_w208.json")
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(output)
    print("public parity:", public_equal)


if __name__ == "__main__":
    main()
