"""Build the first score-conditioned public candidate from scored 67.74.

Increase early cobalt and natural-gas orders by 20% while preserving every
forecast, supplier share, other order, and private row. These two materials
share exposure to the warned Southern shipping lane through Orinoque.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from pipeline import CompetitionData, read_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "submission67.74.csv"
BASE_SHA = "e497bb68ff27835959f8a62dcc21653218725ce9381885295b4a0b686765df0f"
MATERIALS = {"cobalt", "natural_gas"}


def build(output: Path) -> Path:
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("The scored 67.74 base file has changed")
    data = CompetitionData(ROOT / "data" / "snapshots" / "w208")
    header, rows = read_csv(BASE)
    count = 0
    for row in rows:
        week = int(row["id"].split("_")[0][1:])
        material = row["id"].split("_", 1)[1]
        if material in MATERIALS and 209 <= week <= 214 and float(row["order_units"]) > 0:
            row["order_units"] = str(round(float(row["order_units"]) * 1.2))
            count += 1
    if count != 10:
        raise ValueError(f"Expected ten changed positive-order rows, got {count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data.validate_file(output)
    _, generated = read_csv(output)
    originals = {row["id"]: row for row in read_csv(BASE)[1]}
    for row in generated:
        old = originals[row["id"]]
        week = int(row["id"].split("_")[0][1:])
        material = row["id"].split("_", 1)[1]
        expected = material in MATERIALS and 209 <= week <= 214 and float(old["order_units"]) > 0
        diff = [col for col in header if row[col] != old[col]]
        if diff != (["order_units"] if expected else []):
            raise ValueError(f"Unexpected fields changed in {row['id']}: {diff}")
    print(output)
    print("SHA-256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "submissions" / "candidates" / "correlated_lane_buffer_w208.csv")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
