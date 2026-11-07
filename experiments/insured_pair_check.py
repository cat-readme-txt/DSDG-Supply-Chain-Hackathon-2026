"""Compare the reproduced lean policy with the source package's insured policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods" / "lean_60_66" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import insured_hedged, lean_hedged, naive  # noqa: E402

from experiments.lean_share_sweep import ORIGINS


def main() -> None:
    data = load_data(Path("data"))
    cache = Path("/private/tmp/dsdg-lean-share-folds")
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 27)
        base = naive(data.until(origin), weeks)
        for window, info_week, win in (
            ("public", origin, range(origin + 1, origin + 14)),
            ("private", origin + 13, range(origin + 14, origin + 27)),
        ):
            info = data.until(info_week)
            scores = {}
            for name, strategy in (("lean_hedged", lean_hedged),
                                   ("insured_hedged", insured_hedged)):
                path = cache / f"{name}_o{origin}_{window}.csv"
                if path.exists():
                    sub = pd.read_csv(path)
                else:
                    sub = strategy(info, weeks)
                    sub.to_csv(path, index=False)
                scores[name] = score_window(sub, base, data, win).as_dict()
            rows.append({"origin": origin, "window": window, "scores": scores,
                         "better": max(scores, key=lambda name: scores[name]["score"])})
            print(f"finished origin {origin} {window}", flush=True)
    summary = {}
    for label, subset in (("development", [r for r in rows if r["origin"] <= 130]),
                          ("later_holdout", [r for r in rows if r["origin"] >= 156])):
        summary[label] = {
            "lean_mean": sum(r["scores"]["lean_hedged"]["score"] for r in subset) / len(subset),
            "insured_mean": sum(r["scores"]["insured_hedged"]["score"] for r in subset) / len(subset),
            "paired_best_mean": sum(max(s["score"] for s in r["scores"].values()) for r in subset) / len(subset),
            "insured_wins": sum(r["better"] == "insured_hedged" for r in subset),
            "window_count": len(subset),
        }
    out = Path("reports/insured_pair_check_w208.json")
    out.write_text(json.dumps({"origins": ORIGINS, "summary": summary, "folds": rows}, indent=2) + "\n")
    print(out)
    print(summary)


if __name__ == "__main__":
    main()
