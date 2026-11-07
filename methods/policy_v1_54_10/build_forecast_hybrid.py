"""Recreate the scored 50.44 control from its two scored parent CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("submissions/submission44.95.csv"))
    parser.add_argument("--forecasts", type=Path, default=Path("submissions/submission54.10.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    header, base = rows(args.base)
    forecast_header, forecasts = rows(args.forecasts)
    if header != forecast_header or len(base) != len(forecasts):
        raise ValueError("Parent files have different shapes")
    by_id = {row["id"]: row for row in forecasts}
    if len(by_id) != len(base) or set(by_id) != {row["id"] for row in base}:
        raise ValueError("Parent files have different IDs")
    for row in base:
        parent = by_id[row["id"]]
        row["disruption_prob"] = parent["disruption_prob"]
        row["price_forecast"] = parent["price_forecast"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(base)


if __name__ == "__main__":
    main()
