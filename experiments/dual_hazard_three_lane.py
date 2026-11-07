"""Build a week-208 public candidate for a material exposed to two independent hazards.

A material qualifies when two supplier-country forecasts have substantial
near-term hit risk, a third has low risk, and that third costs at most 5% above
the cheapest. Replan the qualifying material with three-supplier near-cost
quotas. Preserve all forecasts and all other rows of the scored 67.74 file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, plan_orders_lp  # noqa: E402
from dsdg_supply_chain.strategies import _with_plan, hedge_forecasts  # noqa: E402
from dsdg_supply_chain.submission import write_submission  # noqa: E402
from pipeline import CompetitionData, read_csv  # noqa: E402

BASE = ROOT / "submissions" / "submission67.74.csv"
BASE_SHA256 = "e497bb68ff27835959f8a62dcc21653218725ce9381885295b4a0b686765df0f"
SHARE_COLS = [f"share_{country}" for country in ("KES", "VAL", "SUN", "ORI", "HAL", "AUR")]


def qualifying_materials(data) -> list[str]:
    feats = build_features(data)
    sup = supplier_forecast(data, range(209, 214), feats=feats)
    state = supplier_state(data, feats, sup)
    result = []
    for material, frame in state.groupby("material"):
        near = frame[frame.week <= 213]
        per_supplier = near.groupby("country").agg(
            risk=("p", "max"), cost=("cost", "mean")).sort_values("risk", ascending=False)
        if len(per_supplier) != 3:
            continue
        risks = per_supplier.risk.to_numpy(float)
        safe_cost = float(per_supplier.cost.iloc[2])
        cheapest = float(per_supplier.cost.min())
        if (risks[0] >= 0.5 and risks[1] >= 0.2 and risks[2] <= 0.05
                and safe_cost <= 1.05 * cheapest):
            result.append(material)
    return sorted(result)


def build(output: Path) -> Path:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    if data.last_week != 208:
        raise ValueError("This candidate is fixed to the week-208 information cutoff")
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("The scored 67.74 base file has changed")
    selected = qualifying_materials(data)
    if selected != ["lithium"]:
        raise ValueError(f"Expected only lithium to satisfy the dual-hazard rule, got {selected}")

    params = LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0,
                      cap_derate=0.96, cheap_premium=0.05, cheap_share=0.30)
    weeks = range(209, 235)
    alternative = hedge_forecasts(
        _with_plan(data, weeks, lambda info, future, feats, sup, material_p:
                   plan_orders_lp(info, future, feats, sup, params)),
        data, "calm")

    with tempfile.TemporaryDirectory(prefix="dsdg-dual-hazard-") as directory:
        planned = Path(directory) / "alternative.csv"
        write_submission(alternative, planned)
        header, base_rows = read_csv(BASE)
        alternative_header, alt_rows = read_csv(planned)
        if header != alternative_header:
            raise ValueError("Alternative header differs from graded base")
        by_id = {row["id"]: row for row in alt_rows}
        changed = 0
        for row in base_rows:
            week = int(row["id"].split("_")[0][1:])
            material = row["id"].split("_", 1)[1]
            if week <= 221 and material in selected:
                replacement = by_id[row["id"]]
                for field in ("order_units", *SHARE_COLS):
                    row[field] = replacement[field]
                changed += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(base_rows)
    CompetitionData(ROOT / "data" / "snapshots" / "w208").validate_file(output)
    _, generated = read_csv(output)
    baseline = {row["id"]: row for row in read_csv(BASE)[1]}
    if len(generated) != len(baseline):
        raise ValueError("Candidate row count changed")
    public_difference = False
    for row in generated:
        original = baseline[row["id"]]
        week = int(row["id"].split("_")[0][1:])
        material = row["id"].split("_", 1)[1]
        if any(row[field] != original[field] for field in ("disruption_prob", "price_forecast")):
            raise ValueError("Candidate forecast columns changed")
        if week > 221 or material not in selected:
            if row != original:
                raise ValueError(f"Unexpected change outside selected public rows: {row['id']}")
        elif row != original:
            public_difference = True
    if not public_difference:
        raise ValueError("Candidate has no distinct public data")
    if changed != 13:
        raise ValueError(f"Expected 13 public lithium rows, got {changed}")
    print("qualifying material:", selected[0])
    print("output:", output)
    print("SHA-256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "generated_candidates" / "dual_hazard_three_lane_w208.csv")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
