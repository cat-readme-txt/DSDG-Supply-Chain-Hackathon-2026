"""Public lithium share-only test against the scored 67.74 order schedule.

Keep every forecast and order quantity. For positive lithium orders in weeks
209–217, use 50% Kestria, 30% Orinoque, and 20% Valdmar. Week 218 keeps its
scored allocation because Kestria's four-week lead misses the window end.
The unchanged week-209 order distinguishes this from the scored 66.98 replan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from pipeline import CompetitionData, read_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "submission67.74.csv"
BASE_SHA256 = "e497bb68ff27835959f8a62dcc21653218725ce9381885295b4a0b686765df0f"
ALLOCATION = {"KES": "0.500000", "ORI": "0.300000", "VAL": "0.200000"}


def build(output: Path) -> Path:
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("The scored 67.74 base file has changed")
    data = CompetitionData(ROOT / "data" / "snapshots" / "w208")
    if set(data.suppliers["lithium"]) != set(ALLOCATION):
        raise ValueError("Unexpected lithium supplier set")
    header, base_rows = read_csv(BASE)
    changes = 0
    for row in base_rows:
        week = int(row["id"].split("_")[0][1:])
        material = row["id"].split("_", 1)[1]
        if 209 <= week <= 217 and material == "lithium" and float(row["order_units"]) > 0:
            before = row.copy()
            for country, target in ALLOCATION.items():
                col = f"share_{country}"
                if float(row[col]) != float(target):
                    row[col] = target
            changes += row != before
    if changes != 9:
        raise ValueError(f"Expected 9 changed positive-order lithium rows, got {changes}")
    data.validate_rows(base_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(base_rows)
    data.validate_file(output)
    _, generated = read_csv(output)
    baseline = {row["id"]: row for row in read_csv(BASE)[1]}
    for row in generated:
        original = baseline[row["id"]]
        if any(row[field] != original[field]
               for field in ("disruption_prob", "price_forecast", "order_units")):
            raise ValueError(f"Unexpected forecast/order change: {row['id']}")
        week = int(row["id"].split("_")[0][1:])
        material = row["id"].split("_", 1)[1]
        if (week > 217 or material != "lithium") and row != original:
            raise ValueError(f"Unexpected row change: {row['id']}")
    print(output)
    print("SHA-256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "generated_candidates" / "lithium_share_only_w208.csv")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
