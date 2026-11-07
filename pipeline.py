"""Reproducible, dependency-free Supply Chain Disruptor candidate generator.

The local simulator follows the published mechanics with the explicit assumptions
documented in PROGRESS.md. It is a backtest tool, not the official scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path


ID_PATTERN = re.compile(r"^w(\d+)_(.+)$")
WINDOWS = ((209, 221), (222, 234))
SHARE_PRECISION = Decimal("0.000001")
LANE_PHRASES = {
    "MERROW": "merrow strait",
    "NORTH": "northern sea route",
    "SOUTH": "southern route",
    "ATLANTIC": "atlantic route",
    "RAIL": "domestic rail",
}
STRONG_WARNING_PHRASES = (
    "flag maintenance delays", "export curbs", "strike date", "strike on",
    "sanctions vote", "prepare for", "traders brace", "accident halts",
    "warning shots", "war risk premiums", "drought", "shortage",
    "closure", "halt exports", "dockworkers",
)
WEAK_WARNING_PHRASES = (
    "congestion builds", "tensions rise", "labor talks drag",
    "debate limits", "slower license approvals", "rush cargo",
)
NEGATED_WARNING_PHRASES = ("averted", "deal reached", "calm returns", "eases", "returns to")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {path}")
        return reader.fieldnames, rows


def parse_id(value: str) -> tuple[int, str]:
    match = ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid submission id: {value}")
    return int(match.group(1)), match.group(2)


def decimal_number(value: str, label: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


@dataclass(frozen=True)
class Profile:
    name: str
    probability: str
    price: str
    shares: str
    order: str = "demand"
    skip: int = 0
    stop: int = 0
    safety: float = 1.0


PROFILES = {
    "cost_reference": Profile("cost_reference", "full", "last", "cheap"),
    "legacy_60_40": Profile("legacy_60_40", "state_decay", "price_blend", "split60"),
    "equal_demand": Profile("equal_demand", "state_decay", "price_blend", "equal"),
    "primary": Profile("primary", "state_decay", "price_blend", "equal", "windowed", 0, 5, 1.2),
    "news_primary": Profile("news_primary", "news_decay", "price_blend", "equal", "windowed", 0, 5, 1.2),
    "cost_lean": Profile("cost_lean", "state_decay", "price_blend", "equal", "windowed", 0, 5, 1.1),
    "shock_hedge": Profile("shock_hedge", "state_decay", "price_blend", "equal", "windowed", 0, 5, 1.3),
}
SUBMISSION_PROFILE_NAMES = ("primary", "news_primary", "cost_lean", "shock_hedge")


class CompetitionData:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.row_counts: dict[str, int] = {}

        def load(name: str) -> tuple[list[str], list[dict[str, str]]]:
            header, rows = read_csv(data_dir / name)
            self.row_counts[name] = len(rows)
            return header, rows

        self.sample_header, self.sample = load("sample_submission.csv")
        self.share_columns = [key for key in self.sample_header if key.startswith("share_")]
        expected_header = [
            "id", "disruption_prob", "price_forecast", "order_units", *self.share_columns
        ]
        if self.sample_header != expected_header or len(self.share_columns) != 6:
            raise ValueError(f"Unexpected sample header: {self.sample_header}")

        _, suppliers = load("suppliers.csv")
        self.suppliers: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in suppliers:
            material, country = row["material"], row["country"]
            if country in self.suppliers[material]:
                raise ValueError(f"Duplicate supplier: {material}/{country}")
            self.suppliers[material][country] = row
        self.materials = tuple(sorted(self.suppliers))
        if len(self.materials) != 10 or any(len(s) != 3 for s in self.suppliers.values()):
            raise ValueError("Expected 10 materials with exactly 3 suppliers each")
        if set(self.share_columns) != {f"share_{c}" for s in self.suppliers.values() for c in s}:
            raise ValueError("Sample share columns do not match supplier countries")

        _, demand = load("demand.csv")
        self.demand: dict[tuple[int, str], int] = {}
        for row in demand:
            key = int(row["week"]), row["material"]
            if key in self.demand:
                raise ValueError(f"Duplicate demand key: {key}")
            quantity = int(row["demand_units"])
            if quantity < 0:
                raise ValueError(f"Negative demand: {key}")
            self.demand[key] = quantity

        _, outcomes = load("outcomes.csv")
        self.outcomes: dict[tuple[int, str], tuple[float, int]] = {}
        for row in outcomes:
            key = int(row["week"]), row["material"]
            if key in self.outcomes:
                raise ValueError(f"Duplicate outcome key: {key}")
            price = float(row["benchmark_price"])
            disrupted = int(row["disrupted"])
            if not math.isfinite(price) or price <= 0 or disrupted not in (0, 1):
                raise ValueError(f"Invalid outcome: {key}")
            self.outcomes[key] = price, disrupted
        self.historical_end = max(week for week, _ in self.outcomes)
        for week in range(1, self.historical_end + 1):
            for material in self.materials:
                if (week, material) not in self.outcomes:
                    raise ValueError(f"Missing historical outcome: {(week, material)}")

        _, supply = load("supply_history.csv")
        self.supply: dict[tuple[int, str, str], tuple[float, int, float]] = {}
        for row in supply:
            key = int(row["week"]), row["material"], row["country"]
            if key in self.supply:
                raise ValueError(f"Duplicate supply key: {key}")
            capacity = float(row["capacity_units"])
            lead = int(row["lead_time_weeks"])
            cost = float(row["landed_cost"])
            if capacity < 0 or lead < 0 or cost <= 0 or not all(
                math.isfinite(x) for x in (capacity, cost)
            ):
                raise ValueError(f"Invalid supply row: {key}")
            self.supply[key] = capacity, lead, cost

        _, prices = load("prices.csv")
        spots: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in prices:
            spots[int(row["week"]), row["material"]].append(float(row["spot_price"]))
        checked = 0
        for key, (benchmark, _) in self.outcomes.items():
            if key in spots:
                if len(spots[key]) != 3 or abs(statistics.mean(spots[key]) - benchmark) > 0.001:
                    raise ValueError(f"Benchmark does not match three spot prices: {key}")
                checked += 1
        self.benchmarks_checked = checked

        for name in ("tariffs.csv", "shipping.csv", "macro.csv"):
            load(name)
        _, news = load("news.csv")
        self.news_by_week: dict[int, list[str]] = defaultdict(list)
        for row in news:
            self.news_by_week[int(row["week"])].append(row["headline"].lower())

        expected_ids = {f"w{week}_{material}" for week in range(209, 235) for material in self.materials}
        actual_ids = [row["id"] for row in self.sample]
        if len(actual_ids) != 260 or set(actual_ids) != expected_ids:
            raise ValueError("Sample does not contain exactly the 260 expected IDs")
        for week in range(209, 235):
            for material in self.materials:
                if (week, material) not in self.demand:
                    raise ValueError(f"Missing future demand: {(week, material)}")
        self.validate_rows(self.sample)

    def audit(self) -> dict[str, object]:
        return {
            "historical_end": self.historical_end,
            "material_count": len(self.materials),
            "supplier_count": sum(len(x) for x in self.suppliers.values()),
            "submission_rows": len(self.sample),
            "benchmark_rows_checked": self.benchmarks_checked,
            "file_rows": self.row_counts,
        }

    def validate_rows(self, rows: list[dict[str, str]]) -> None:
        if len(rows) != len(self.sample):
            raise ValueError(f"Expected {len(self.sample)} rows, got {len(rows)}")
        expected = {row["id"] for row in self.sample}
        seen: set[str] = set()
        for index, row in enumerate(rows, 2):
            if set(row) != set(self.sample_header):
                raise ValueError(f"Row {index}: wrong columns")
            identifier = row["id"]
            week, material = parse_id(identifier)
            if identifier not in expected or identifier in seen:
                raise ValueError(f"Row {index}: missing, unexpected, or duplicate ID {identifier}")
            seen.add(identifier)
            probability = decimal_number(row["disruption_prob"], f"row {index} probability")
            price = decimal_number(row["price_forecast"], f"row {index} price")
            units = decimal_number(row["order_units"], f"row {index} units")
            if not Decimal(0) <= probability <= Decimal(1) or price <= 0 or units < 0:
                raise ValueError(f"Row {index}: invalid probability, price, or units")
            shares = {
                key.removeprefix("share_"): decimal_number(row[key], f"row {index} {key}")
                for key in self.share_columns
            }
            if any(not Decimal(0) <= value <= Decimal(1) for value in shares.values()):
                raise ValueError(f"Row {index}: share outside [0, 1]")
            if abs(sum(shares.values()) - Decimal(1)) > SHARE_PRECISION:
                raise ValueError(f"Row {index}: shares do not sum to 1")
            if any(value != 0 and country not in self.suppliers[material] for country, value in shares.items()):
                raise ValueError(f"Row {index}: share for ineligible country")
            if (week, material) not in self.demand:
                raise ValueError(f"Row {index}: unknown material-week")
        if seen != expected:
            raise ValueError("Submission ID set does not match sample")

    def validate_file(self, path: Path) -> None:
        header, rows = read_csv(path)
        if header != self.sample_header:
            raise ValueError("Submission header differs from sample")
        self.validate_rows(rows)

    @lru_cache(maxsize=None)
    def warning_score(self, material: str, origin: int) -> float:
        """Recent relevant headlines, using only articles available by origin."""
        material_phrase = material.replace("_", " ")
        suppliers = self.suppliers[material].values()
        country_phrases = [row["country_name"].lower() for row in suppliers]
        lane_phrases = [LANE_PHRASES[row["lane"]] for row in suppliers]
        score = 0.0
        for week in range(max(1, origin - 1), origin + 1):
            for headline in self.news_by_week.get(week, []):
                if any(phrase in headline for phrase in NEGATED_WARNING_PHRASES):
                    continue
                material_match = material_phrase in headline or (
                    material_phrase.endswith("s") and material_phrase[:-1] in headline
                )
                if any(other.replace("_", " ") in headline for other in self.materials if other != material):
                    if not material_match:
                        continue
                if not (
                    material_match
                    or any(country in headline for country in country_phrases)
                    or any(lane in headline for lane in lane_phrases)
                ):
                    continue
                if any(phrase in headline for phrase in STRONG_WARNING_PHRASES):
                    strength = 1.0
                elif any(phrase in headline for phrase in WEAK_WARNING_PHRASES):
                    strength = 0.4
                else:
                    continue
                score += strength * 0.85 ** (origin - week)
        return min(1.0, score)

    def probability(self, material: str, origin: int, method: str, horizon: int) -> float:
        history = [self.outcomes[week, material][1] for week in range(1, origin + 1)]
        base = statistics.mean(history)
        if method == "full":
            value = base
        elif method == "blend52":
            value = 0.5 * (base + statistics.mean(history[-52:]))
        elif method == "state_decay":
            value = base + (history[-1] - base) * (0.8 ** horizon)
        elif method == "news_decay":
            state_value = base + (history[-1] - base) * (0.8 ** horizon)
            warning = self.warning_score(material, origin)
            value = state_value + (1 - state_value) * 0.6 * warning * (0.95 ** horizon)
        else:
            raise ValueError(f"Unknown probability method: {method}")
        return min(1.0, max(0.0, value))

    def price(self, material: str, origin: int, method: str) -> float:
        history = [self.outcomes[week, material][0] for week in range(1, origin + 1)]
        last = history[-1]
        if method == "last":
            value = last
        elif method == "mean4":
            value = statistics.mean(history[-4:])
        elif method == "price_blend":
            value = last + 0.25 * (statistics.mean(history[-4:]) - last)
        else:
            raise ValueError(f"Unknown price method: {method}")
        return max(0.0001, value)

    @lru_cache(maxsize=None)
    def ranked_suppliers(self, material: str, origin: int) -> list[str]:
        costs = {}
        for country in self.suppliers[material]:
            observed = sorted(
                (week, value[2])
                for (week, mat, c), value in self.supply.items()
                if mat == material and c == country and week <= origin
            )
            if not observed:
                raise ValueError(f"No cost history for {material}/{country} through week {origin}")
            costs[country] = statistics.mean(value for _, value in observed[-26:])
        return sorted(costs, key=lambda country: (costs[country], country))

    @lru_cache(maxsize=None)
    def shares(self, material: str, origin: int, method: str) -> dict[str, Decimal]:
        ranking = self.ranked_suppliers(material, origin)
        result = {country: Decimal(0) for country in self.suppliers[material]}
        weights = {
            "cheap": ("1", "0", "0"),
            "split60": ("0.6", "0.4", "0"),
            "split50": ("0.5", "0.25", "0.25"),
            "split45": ("0.45", "0.30", "0.25"),
            "equal": ("0.333333", "0.333333", "0.333334"),
        }.get(method)
        if weights is None:
            raise ValueError(f"Unknown share method: {method}")
        for country, weight in zip(ranking, weights):
            result[country] = Decimal(weight)
        return result

    @staticmethod
    def window_for_week(week: int) -> tuple[int, int]:
        for start, end in WINDOWS:
            if start <= week <= end:
                return start, end
        raise ValueError(f"Week {week} outside submission windows")

    def origin_for_window(self, start: int) -> int:
        return min(self.historical_end, start - 1)

    def policy_units(self, material: str, week: int, start: int, end: int, origin: int, profile: Profile) -> int:
        if profile.order == "demand":
            return self.demand[week, material]
        if profile.order in ("windowed", "risk_scaled"):
            offset = week - start
            if offset < profile.skip or offset >= (end - start + 1 - profile.stop):
                return 0
            if profile.order == "windowed":
                multiplier = profile.safety
            else:
                historical_rate = statistics.mean(self.outcomes[past_week, material][1] for past_week in range(1, origin + 1))
                multiplier = 1 + profile.safety * historical_rate
            return max(0, round(self.demand[week, material] * multiplier))
        raise ValueError(f"Unknown order method: {profile.order}")

    def order_units(self, material: str, week: int, profile: Profile) -> int:
        start, end = self.window_for_week(week)
        origin = self.origin_for_window(start)
        return self.policy_units(material, week, start, end, origin, profile)

    def candidate_rows(self, profile: Profile) -> list[dict[str, str]]:
        result = []
        for template in self.sample:
            week, material = parse_id(template["id"])
            start, _ = self.window_for_week(week)
            origin = self.origin_for_window(start)
            allocation = self.shares(material, origin, profile.shares)
            row = {key: "0" for key in self.sample_header}
            row["id"] = template["id"]
            row["disruption_prob"] = f"{self.probability(material, origin, profile.probability, week-origin):.6f}"
            row["price_forecast"] = f"{self.price(material, origin, profile.price):.4f}"
            row["order_units"] = str(self.order_units(material, week, profile))
            for country, value in allocation.items():
                row[f"share_{country}"] = f"{value:.6f}"
            result.append(row)
        self.validate_rows(result)
        return result

    def write_candidate(self, profile: Profile, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{profile.name}_w{self.historical_end}.csv"
        rows = self.candidate_rows(profile)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.sample_header)
            writer.writeheader()
            writer.writerows(rows)
        self.validate_file(path)
        return path

    def complete_fold_origins(self) -> list[int]:
        """13-week folds available after a minimum of 52 training weeks."""
        return list(range(52, self.historical_end - 12, 13))

    def forecast_backtest(self) -> dict[str, object]:
        origins = self.complete_fold_origins()
        methods = ("full", "blend52", "state_decay", "news_decay")
        price_methods = ("last", "mean4", "price_blend")
        values: dict[str, list[float]] = defaultdict(list)
        folds = []
        for origin in origins:
            local: dict[str, list[float]] = defaultdict(list)
            for material in self.materials:
                for week in range(origin + 1, origin + 14):
                    actual_price, actual_event = self.outcomes[week, material]
                    horizon = week - origin
                    for method in methods:
                        error = (self.probability(material, origin, method, horizon) - actual_event) ** 2
                        values[f"brier_{method}"].append(error)
                        local[f"brier_{method}"].append(error)
                    for method in price_methods:
                        error = abs(self.price(material, origin, method) - actual_price) / actual_price
                        values[f"mape_{method}"].append(error)
                        local[f"mape_{method}"].append(error)
            folds.append({
                "weeks": f"{origin+1}-{origin+13}",
                "brier_full": statistics.mean(local["brier_full"]),
                "brier_state_decay": statistics.mean(local["brier_state_decay"]),
                "brier_news_decay": statistics.mean(local["brier_news_decay"]),
                "mape_last": statistics.mean(local["mape_last"]),
                "mape_price_blend": statistics.mean(local["mape_price_blend"]),
            })
        summary = {key: statistics.mean(items) for key, items in values.items()}
        summary["brier_skill_state_decay_vs_full"] = 1 - summary["brier_state_decay"] / summary["brier_full"]
        summary["brier_skill_news_decay_vs_full"] = 1 - summary["brier_news_decay"] / summary["brier_full"]
        summary["price_skill_blend_vs_last"] = 1 - summary["mape_price_blend"] / summary["mape_last"]
        return {"fold_count": len(folds), "predictions": len(values["brier_full"]), "summary": summary, "folds": folds}

    def simulate_window(
        self,
        start: int,
        end: int,
        origin: int,
        profile: Profile,
        initial_stock: str = "sum_first_five",
    ) -> dict[str, float]:
        total = defaultdict(float)
        for material in self.materials:
            if initial_stock == "sum_first_five":
                inventory = float(sum(self.demand[week, material] for week in range(start, min(end, start + 4) + 1)))
            elif initial_stock == "five_times_first":
                inventory = float(5 * self.demand[start, material])
            else:
                raise ValueError(f"Unknown initial stock assumption: {initial_stock}")
            arrival: dict[int, float] = defaultdict(float)
            in_transit = 0.0
            allocation = self.shares(material, origin, profile.shares)
            avg_costs = [
                statistics.mean(self.supply[week, material, country][2] for country in self.suppliers[material])
                for week in range(start, end + 1)
            ]
            salvage_price = statistics.mean(avg_costs)
            for offset, week in enumerate(range(start, end + 1)):
                demand = self.demand[week, material]
                units = self.policy_units(material, week, start, end, origin, profile)
                for country, share in allocation.items():
                    requested = units * float(share)
                    capacity, lead, landed = self.supply[week, material, country]
                    filled = min(requested, capacity)
                    total["purchase_cost"] += filled * landed
                    total["ordered_units"] += requested
                    total["filled_order_units"] += filled
                    arrival_week = week + lead
                    if arrival_week <= end:
                        arrival[arrival_week] += filled
                    else:
                        in_transit += filled
                inventory += arrival[week]
                met = min(inventory, demand)
                inventory -= met
                shortage = demand - met
                total["demand_units"] += demand
                total["met_units"] += met
                total["shortage_units"] += shortage
                total["holding_cost"] += 0.02 * avg_costs[offset] * inventory
                total["shortage_cost"] += 5 * avg_costs[offset] * shortage
                if any(self.outcomes[shock_week, material][1] for shock_week in range(max(1, week-4), week+1)):
                    total["shock_demand_units"] += demand
                    total["shock_met_units"] += met
            total["salvage_value"] += 0.70 * salvage_price * (inventory + in_transit)
            total["ending_stock_units"] += inventory
            total["ending_transit_units"] += in_transit
            total["diversity_sum"] += 1.5 * (1 - sum(float(value) ** 2 for value in allocation.values()))
        total.setdefault("shock_met_units", 0.0)
        total.setdefault("shock_demand_units", 0.0)
        total["total_cost"] = total["purchase_cost"] + total["holding_cost"] + total["shortage_cost"] - total["salvage_value"]
        total["fill_rate"] = total["met_units"] / total["demand_units"]
        total["shock_fill_rate"] = total["shock_met_units"] / total["shock_demand_units"] if total["shock_demand_units"] else None
        total["diversity"] = total["diversity_sum"] / len(self.materials)
        del total["diversity_sum"]
        return dict(total)

    def simulator_backtest(self, profile: Profile, initial_stock: str = "sum_first_five") -> dict[str, object]:
        folds = []
        for origin in self.complete_fold_origins():
            folds.append({
                "weeks": f"{origin+1}-{origin+13}",
                **self.simulate_window(origin + 1, origin + 13, origin, profile, initial_stock),
            })
        return {"profile": profile.name, "initial_stock": initial_stock, "folds": folds}

    def policy_search(self) -> dict[str, object]:
        """Train on the first eight 13-week folds and reserve the last four as holdout."""
        origins = self.complete_fold_origins()
        if len(origins) < 12:
            raise ValueError("Policy search requires at least twelve complete 13-week historical folds")

        def summarize(folds: list[dict[str, float]]) -> dict[str, float]:
            shock_demand = sum(fold["shock_demand_units"] for fold in folds)
            return {
                "cost": sum(fold["total_cost"] for fold in folds),
                "fill_rate": sum(fold["met_units"] for fold in folds) / sum(fold["demand_units"] for fold in folds),
                "shock_fill_rate": sum(fold["shock_met_units"] for fold in folds) / shock_demand if shock_demand else 1.0,
                "diversity": statistics.mean(fold["diversity"] for fold in folds),
            }

        trials = []
        for share_method in ("cheap", "split60", "split50", "split45", "equal"):
            for skip in (0, 1, 2):
                for stop in (2, 3, 4, 5):
                    for safety in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
                        profile = Profile("trial", "state_decay", "price_blend", share_method, "windowed", skip, stop, safety)
                        folds = [self.simulate_window(origin+1, origin+13, origin, profile) for origin in origins]
                        trials.append({
                            "shares": share_method,
                            "skip": skip,
                            "stop": stop,
                            "safety": safety,
                            "training": summarize(folds[:8]),
                            "holdout": summarize(folds[8:]),
                        })
        eligible = [trial for trial in trials if trial["training"]["shock_fill_rate"] >= 0.99]
        eligible.sort(key=lambda trial: trial["training"]["cost"])
        return {
            "trial_count": len(trials),
            "training_weeks": "53-156",
            "holdout_weeks": f"157-{origins[-1]+13}",
            "selection_gate": "training shock fill >= 0.99, then lowest training cost",
            "eligible_count": len(eligible),
            "top_eligible": eligible[:15],
            "selected_profiles": {
                name: next(
                    (trial for trial in trials if trial["shares"] == PROFILES[name].shares
                     and trial["skip"] == PROFILES[name].skip
                     and trial["stop"] == PROFILES[name].stop
                     and trial["safety"] == PROFILES[name].safety),
                    None,
                )
                for name in SUBMISSION_PROFILE_NAMES
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "backtest", "search", "generate", "validate", "all"))
    parser.add_argument("path", nargs="?", type=Path, help="CSV path for validate")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    data = CompetitionData(args.data_dir)
    if args.command in ("audit", "all"):
        print(json.dumps({"audit": data.audit()}, indent=2))
    if args.command in ("backtest", "all"):
        report = {
            "data_week": data.historical_end,
            "forecast": data.forecast_backtest(),
            "simulator_assumptions": [
                "Initial stock equals sum of first five weekly demands per material",
                "Orders placed in week t arrive in week t plus observed lead time",
                "Holding charged on inventory remaining after weekly demand",
                "Average landed cost is the same-week mean across three suppliers",
                "Ending stock and transit salvaged at 70% of window mean landed cost",
            ],
            "simulator": {
                name: data.simulator_backtest(profile)
                for name, profile in PROFILES.items()
            },
        }
        args.report_dir.mkdir(parents=True, exist_ok=True)
        path = args.report_dir / f"backtest_w{data.historical_end}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"backtest_report": str(path), "forecast_summary": report["forecast"]["summary"]}, indent=2))
    if args.command in ("search", "all"):
        search = data.policy_search()
        args.report_dir.mkdir(parents=True, exist_ok=True)
        path = args.report_dir / f"policy_search_w{data.historical_end}.json"
        path.write_text(json.dumps(search, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"policy_search_report": str(path), "trial_count": search["trial_count"], "top_training_policy": search["top_eligible"][0]}, indent=2))
    if args.command in ("generate", "all"):
        files = {name: str(data.write_candidate(PROFILES[name], args.output_dir)) for name in SUBMISSION_PROFILE_NAMES}
        print(json.dumps({"generated_and_validated": files}, indent=2))
    if args.command == "validate":
        if args.path is None:
            parser.error("validate requires a CSV path")
        data.validate_file(args.path)
        print(json.dumps({"valid": str(args.path), "rows": len(data.sample)}))


if __name__ == "__main__":
    main()
