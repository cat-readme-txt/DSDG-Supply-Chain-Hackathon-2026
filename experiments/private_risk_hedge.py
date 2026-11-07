"""Build a private-window hedge from a scored submission without changing public rows."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

from pipeline import CompetitionData, SHARE_PRECISION, parse_id, read_csv


def build(base: Path, output: Path, data: CompetitionData,
          slope: Decimal = Decimal("3"), cap: Decimal = Decimal("0.6")) -> Path:
    header, source = read_csv(base)
    if header != data.sample_header:
        raise ValueError("Base submission header differs from the template")
    data.validate_rows(source)
    result = []
    changed = 0
    for original in source:
        row = original.copy()
        week, material = parse_id(row["id"])
        if week >= 222 and Decimal(row["order_units"]) > 0:
            delta = min(cap, slope * Decimal(row["disruption_prob"]))
            countries = tuple(data.suppliers[material])
            allocation = {}
            for country in countries[:2]:
                weight = Decimal(row[f"share_{country}"])
                allocation[country] = ((Decimal(1)-delta)*weight + delta/3).quantize(SHARE_PRECISION)
            allocation[countries[2]] = Decimal(1) - sum(allocation.values())
            for country, value in allocation.items():
                row[f"share_{country}"] = f"{value:.6f}"
            changed += 1
        result.append(row)
    data.validate_rows(result)
    if changed == 0 or any(a != b for a, b in zip(source, result)
                           if parse_id(a["id"])[0] <= 221):
        raise ValueError("Expected private share changes and exact public-row parity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(result)
    data.validate_file(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("submissions/submission60.66.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/candidates/private_risk_hedge_w208.csv"))
    args = parser.parse_args()
    data = CompetitionData(Path("data"))
    print(build(args.base, args.output, data))


if __name__ == "__main__":
    main()
