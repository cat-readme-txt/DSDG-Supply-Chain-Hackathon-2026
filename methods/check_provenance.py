"""Check graded CSVs and the selected final pair against archived source.

The first fifteen files reproduce byte for byte. The 92.62 public optimization
has small numerical drift with the current source/runtime; its private rows and
all forecasts still match exactly, checked separately below.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "snapshots" / "w208"
PHASE2 = ROOT / "data" / "Phase 2 data"
SCORED = ROOT / "submissions"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, score: str) -> None:
    expected = SCORED / f"submission{score}.csv"
    if path.read_bytes() != expected.read_bytes():
        raise AssertionError(f"{path} does not reproduce {expected}")
    print(f"{score}: exact match ({digest(path)[:16]})")


def check_joint_drift(path: Path) -> None:
    import pandas as pd

    expected = pd.read_csv(SCORED / "submission92.62.csv").set_index("id").sort_index()
    generated = pd.read_csv(path).set_index("id").sort_index()
    if not expected.index.equals(generated.index):
        raise AssertionError("92.62 IDs differ")
    private = expected.index.to_series().str.extract(r"w(\d+)_")[0].astype(int) >= 222
    if not expected.loc[private].equals(generated.loc[private]):
        raise AssertionError("92.62 private rows differ")
    for col in ("disruption_prob", "price_forecast"):
        if not expected[col].equals(generated[col]):
            raise AssertionError(f"92.62 forecast field {col} differs")
    public_order_delta = (expected.loc[~private, "order_units"] -
                          generated.loc[~private, "order_units"]).abs()
    shares = [c for c in expected if c.startswith("share_")]
    public_share_delta = (expected.loc[~private, shares] - generated.loc[~private, shares]).abs()
    if int((public_order_delta > 0).sum()) != 2 or public_order_delta.max() > 2:
        raise AssertionError("92.62 public order drift exceeded observed bounds")
    if public_share_delta.to_numpy().max() > 0.001:
        raise AssertionError("92.62 public share drift exceeded observed bounds")
    print("92.62: source match with public optimization drift (private and forecasts exact; "
          "2 public orders differ by at most 2 units; shares by at most 0.001)")


def check_final_pair(python: str, temporary: Path) -> None:
    """Regenerate the selected pair without changing the submission artifacts."""
    output = temporary / "final_pair"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run([python, "-m", "experiments.private_closure_conditional",
         "--alpha", "1", "--output", str(output)], env)
    for side in ("lean", "insured"):
        name = f"private_closure_conditional_{side}_w221.csv"
        expected = SCORED / "candidates" / name
        generated = output / name
        if generated.read_bytes() != expected.read_bytes():
            raise AssertionError(f"{name} does not match its saved final artifact")
        print(f"final {side}: exact match ({digest(generated)[:16]})")


def build_strategy(python: str, source: Path, name: str, output: Path) -> None:
    env = os.environ.copy()
    env.update(DSDG_DATA_DIR=str(DATA), PYTHONPATH=str(source / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    code = 'import sys; from dsdg_supply_chain.cli import main; main(["submit", sys.argv[1], "-o", sys.argv[2]])'
    run([python, "-c", code, name, str(output)], env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable with numpy, pandas, scipy, scikit-learn")
    parser.add_argument("--final-only", action="store_true",
                        help="regenerate and compare only the selected final pair")
    args = parser.parse_args()
    for name in ("demand", "macro", "news", "outcomes", "prices", "shipping",
                 "suppliers", "supply_history", "tariffs", "sample_submission"):
        if not (DATA / f"{name}.csv").exists():
            raise FileNotFoundError(DATA / f"{name}.csv")
        if not (PHASE2 / f"{name}.csv").exists():
            raise FileNotFoundError(PHASE2 / f"{name}.csv")
    with tempfile.TemporaryDirectory(prefix="dsdg-provenance-") as directory:
        temp = Path(directory)
        if args.final_only:
            check_final_pair(args.python, temp)
            return
        run([args.python, str(ROOT / "pipeline.py"), "generate", "--data-dir", str(DATA),
             "--output-dir", str(temp / "pipeline")])
        for score, profile in (("44.95", "primary"), ("45.80", "lead_aware"),
                               ("46.21", "long_memory"), ("46.22", "independent")):
            check(temp / "pipeline" / f"{profile}_w208.csv", score)

        old = ROOT / "methods" / "policy_v1_54_10"
        build_strategy(args.python, old, "policy_v1", temp / "policy_v1.csv")
        check(temp / "policy_v1.csv", "54.10")
        run([args.python, str(old / "build_forecast_hybrid.py"), "--base",
             str(SCORED / "submission44.95.csv"), "--forecasts",
             str(SCORED / "submission54.10.csv"), "--out", str(temp / "hybrid.csv")])
        check(temp / "hybrid.csv", "50.44")

        lean = ROOT / "methods" / "lean_60_66"
        for score, strategy in (("48.55", "insured_hedged"), ("60.66", "lean_hedged")):
            output = temp / f"{strategy}.csv"
            build_strategy(args.python, lean, strategy, output)
            check(output, score)

        split = ROOT / "methods" / "lean_split_67_74"
        build_strategy(args.python, split, "lean_split", temp / "lean_split.csv")
        check(temp / "lean_split.csv", "67.74")
        run([args.python, "-m", "experiments.dual_hazard_three_lane", "--output",
             str(temp / "dual_hazard.csv")])
        check(temp / "dual_hazard.csv", "66.98")
        run([args.python, "-m", "experiments.lithium_share_only", "--output",
             str(temp / "lithium_share.csv")])
        check(temp / "lithium_share.csv", "67.26")
        run([args.python, "-m", "experiments.phase2_public_oracle", "--convention",
             "sum5", "--output", str(temp / "phase2_oracle.csv")])
        check(temp / "phase2_oracle.csv", "89.45")
        run([args.python, "-m", "experiments.phase2_public_oracle", "--convention",
             "5mean_supweek", "--output", str(temp / "phase2_5mean.csv")])
        check(temp / "phase2_5mean.csv", "84.27")
        run([args.python, "-m", "experiments.public_diversification_qp", "--diversity-scale",
             "0.5", "--output", str(temp / "phase2_qp.csv")])
        check(temp / "phase2_qp.csv", "91.36")
        run([args.python, "-m", "experiments.public_diversification_qp", "--diversity-scale",
             "1.5", "--output", str(temp / "phase2_qp_1_5.csv")])
        check(temp / "phase2_qp_1_5.csv", "92.31")
        run([args.python, "-m", "experiments.public_joint_order_search", "--base",
             str(SCORED / "submission92.31.csv"), "--scale", "1.5", "--output",
             str(temp / "phase2_joint.csv")])
        check_joint_drift(temp / "phase2_joint.csv")
        check_final_pair(args.python, temp)


if __name__ == "__main__":
    main()
