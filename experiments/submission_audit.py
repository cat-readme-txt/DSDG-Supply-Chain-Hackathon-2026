"""Validate every stored submission and find redundant full or public CSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from pipeline import CompetitionData, parse_id, read_csv


def canonical_rows(path: Path, data: CompetitionData) -> tuple[tuple[str, tuple[Decimal, ...]], ...]:
    data.validate_file(path)
    _, rows = read_csv(path)
    numeric_columns = [column for column in data.sample_header if column != "id"]
    return tuple(sorted(
        (row["id"], tuple(Decimal(row[column]) for column in numeric_columns))
        for row in rows
    ))


def duplicate_groups(signatures: dict[Path, tuple]) -> list[list[str]]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for path, signature in signatures.items():
        groups[signature].append(str(path))
    return [sorted(paths) for paths in groups.values() if len(paths) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("submissions"))
    args = parser.parse_args()
    data = CompetitionData(Path("data"))
    paths = sorted(args.root.rglob("*.csv"))
    signatures = {path: canonical_rows(path, data) for path in paths}
    public = {
        path: tuple(row for row in signature if parse_id(row[0])[0] <= 221)
        for path, signature in signatures.items()
    }
    print(f"Validated {len(paths)} CSVs ({len(data.sample)} rows each)")
    print("Full-data duplicate groups:", duplicate_groups(signatures))
    print("Public-week duplicate groups:", duplicate_groups(public))


if __name__ == "__main__":
    main()
