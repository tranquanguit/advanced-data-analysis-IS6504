"""Unified runner: non-linear analysis + all 6 forecasting scenarios.

Usage:
    python run_hybrid.py                        # run everything
    python run_hybrid.py --skip-nonlinear       # skip non-linear, only run 6 scenarios
    python run_hybrid.py --skip-scenarios       # skip scenarios, only run non-linear
    python run_hybrid.py --scenarios 1 2        # run only scenario 1 and 2
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

CONFIG_PATH = "configs/default.yaml"
OUTPUTS_DIR = "outputs"
RESULTS_DIR = "results"

SCENARIOS = [
    {
        "id": "scenario1",
        "desc": "Dengue - climate + social only",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False,
    },
    {
        "id": "scenario2",
        "desc": "Dengue - climate + social + cross-disease features",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": True,
    },
    {
        "id": "scenario3",
        "desc": "Influenza - climate + social only",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False,
    },
    {
        "id": "scenario4",
        "desc": "Influenza - climate + social + cross-disease features",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": True,
    },
    {
        "id": "scenario5",
        "desc": "Diarrhoea - climate + social only",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False,
    },
    {
        "id": "scenario6",
        "desc": "Diarrhoea - climate + social + cross-disease features",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full hybrid pipeline: non-linear analysis + all 6 forecasting scenarios",
    )
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--skip-nonlinear", action="store_true", help="Skip non-linear analysis step")
    parser.add_argument("--skip-scenarios", action="store_true", help="Skip all forecasting scenarios")
    parser.add_argument("--scenarios", nargs="+", type=int, metavar="N",
                        help="Run only specific scenario numbers (e.g. --scenarios 1 2 5)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_config(config_path: str, scenario: dict) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["experiment"]["target"] = scenario["target"]
    config["experiment"]["cases_col"] = scenario["cases_col"]
    config["experiment"]["compute_rate_per100k"] = scenario["compute_rate_per100k"]
    config["experiment"]["include_other_diseases_as_features"] = scenario["include_other_diseases_as_features"]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def copy_results(scenario_id: str) -> None:
    dest_dir = os.path.join(RESULTS_DIR, scenario_id, OUTPUTS_DIR)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(os.path.join(RESULTS_DIR, scenario_id), exist_ok=True)
    shutil.copytree(OUTPUTS_DIR, dest_dir)


def run_cmd(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config_path = args.config
    root_dir = Path(__file__).resolve().parent
    start_time = datetime.now()

    print(f"{'='*60}")
    print(f"  HYBRID PIPELINE - started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}")

    step_errors: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Non-linear correlation analysis
    # ------------------------------------------------------------------
    if not args.skip_nonlinear:
        print(f"\n{'-'*60}")
        print("  STEP 1/2: Non-linear correlation analysis")
        print(f"{'-'*60}")
        try:
            run_cmd([sys.executable, "run_nonlinear.py", "--config", config_path])
            print("[OK] Non-linear analysis completed.")
        except subprocess.CalledProcessError as exc:
            msg = f"Non-linear analysis failed: {exc}"
            step_errors.append(msg)
            print(f"[WARN] {msg}")
    else:
        print("\n[SKIP] Non-linear analysis (--skip-nonlinear)")

    # ------------------------------------------------------------------
    # Step 2: Forecasting scenarios (6 scenarios)
    # ------------------------------------------------------------------
    if not args.skip_scenarios:
        # Determine which scenarios to run
        if args.scenarios:
            selected = [s for s in SCENARIOS if int(s["id"].replace("scenario", "")) in args.scenarios]
        else:
            selected = SCENARIOS

        print(f"\n{'-'*60}")
        print(f"  STEP 2/2: Forecasting - {len(selected)} scenario(s)")
        print(f"{'-'*60}")

        for i, scenario in enumerate(selected, 1):
            sid = scenario["id"]
            print(f"\n  >> [{i}/{len(selected)}] {sid}: {scenario['desc']}")
            update_config(config_path, scenario)
            try:
                run_cmd([sys.executable, "run_all.py", "--config", config_path])

                copy_results(sid)
                print(f"  >> [OK] {sid} completed.")
            except Exception as exc:
                msg = f"{sid} failed: {exc}"
                step_errors.append(msg)
                print(f"  >> [FAIL] {msg}")
    else:
        print("\n[SKIP] Forecasting scenarios (--skip-scenarios)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    if step_errors:
        print(f"  DONE with {len(step_errors)} error(s) in {elapsed}")
        for err in step_errors:
            print(f"    [!] {err}")
    else:
        print(f"  DONE - all steps completed in {elapsed}")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
