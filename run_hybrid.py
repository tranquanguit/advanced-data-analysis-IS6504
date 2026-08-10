"""Unified runner: non-linear analysis + all 6 forecasting scenarios.

Usage:
    python run_hybrid.py                        # run everything
    python run_hybrid.py --skip-nonlinear       # skip non-linear, only run 6 scenarios
    python run_hybrid.py --skip-scenarios       # skip scenarios, only run non-linear
    python run_hybrid.py --scenarios 1 2        # run only scenario 1 and 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
        "cross_disease_map": None,
    },
    {
        "id": "scenario2",
        "desc": "Dengue - climate + social + cross-disease features (all lags)",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": {
            "Influenza_rates": list(range(1, 13)),
            "Diarrhoea_rates": list(range(1, 13))
        },
    },
    {
        "id": "scenario3",
        "desc": "Influenza - climate + social only",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": None,
    },
    {
        "id": "scenario4",
        "desc": "Influenza - climate + social + cross-disease features (all lags)",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": {
            "Dengue_fever_rates": list(range(1, 13)),
            "Diarrhoea_rates": list(range(1, 13))
        },
    },
    {
        "id": "scenario5",
        "desc": "Diarrhoea - climate + social only",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": None,
    },
    {
        "id": "scenario6",
        "desc": "Diarrhoea - climate + social + cross-disease features (all lags)",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": {
            "Dengue_fever_rates": list(range(1, 13)),
            "Influenza_rates": list(range(1, 13))
        },
    },
    {
        "id": "scenario7",
        "desc": "Dengue - NL-guided cross-disease (no signal)",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": None,
    },
    {
        "id": "scenario8",
        "desc": "Influenza - NL-guided cross-disease (Diarrhoea lag 0-3)",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": {
            "Diarrhoea_rates": [0, 1, 2, 3]
        },
    },
    {
        "id": "scenario9",
        "desc": "Diarrhoea - NL-guided cross-disease (Influenza lag 0-1)",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "cross_disease_map": {
            "Influenza_rates": [0, 1]
        },
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
    config["experiment"]["cross_disease_map"] = scenario["cross_disease_map"]

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


def load_suggested_lags(json_path: Path) -> dict:
    """Load suggested lags from NL analysis Step 1."""
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load suggested lags: {e}")
        return {}


def cross_scenario_wilcoxon(base_id: str, cross_id: str, results_dir: str) -> None:
    """Run Wilcoxon signed-rank test between a base scenario and a cross-disease scenario."""
    models = ["HistGB", "XGBoost", "LSTM", "LightGBM"]
    horizons = [1, 2, 3, 4, 5, 6]
    
    rows = []
    print(f"\n  >> Statistical Test: {base_id} (Base) vs {cross_id} (Cross)")
    for model in models:
        base_path = Path(results_dir) / base_id / "outputs" / "predictions" / f"pred_{model}.csv"
        cross_path = Path(results_dir) / cross_id / "outputs" / "predictions" / f"pred_{model}.csv"
        
        if not base_path.exists() or not cross_path.exists():
            continue
            
        df_b = pd.read_csv(base_path)
        df_c = pd.read_csv(cross_path)
        
        for h in horizons:
            err_b = np.abs(df_b[f"actual_t+{h}"] - df_b[f"pred_t+{h}"])
            err_c = np.abs(df_c[f"actual_t+{h}"] - df_c[f"pred_t+{h}"])
            
            if np.allclose(err_b, err_c):
                p_val = 1.0 # Identical outputs
            else:
                try:
                    stat, p_val = wilcoxon(err_b, err_c)
                except ValueError:
                    p_val = 1.0
            
            mean_b = err_b.mean()
            mean_c = err_c.mean()
            delta = (mean_c - mean_b) / mean_b * 100 if mean_b > 0 else 0
            
            rows.append({
                "model": model,
                "horizon": f"MAE@{h}",
                "base_mae": mean_b,
                "cross_mae": mean_c,
                "delta_%": delta,
                "p_value": p_val,
                "significant": p_val < 0.05
            })
            
    if rows:
        out_df = pd.DataFrame(rows)
        out_file = Path(results_dir) / cross_id / "outputs" / "metrics" / f"wilcoxon_vs_{base_id}.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_file, index=False)
        # Display subset 
        sub = out_df[out_df["horizon"].isin(["MAE@1", "MAE@3", "MAE@6"])]
        print(f"     Results saved to {out_file.name}")



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
        # Load suggested lags if available
        suggested_path = Path(OUTPUTS_DIR) / "nonlinear" / "tables" / "suggested_lags.json"
        dynamic_lags = load_suggested_lags(suggested_path)
        
        if dynamic_lags:
            print(f"[INFO] Dynamically updating Scenarios 7, 8, 9 with NL results...")
            for s in SCENARIOS:
                sid = s["id"]
                target = s["target"]
                if sid in ["scenario7", "scenario8", "scenario9"]:
                    lags_for_target = dynamic_lags.get(target, {})
                    # Only keep predictors that have non-empty lag lists
                    filtered_lags = {p: l for p, l in lags_for_target.items() if l}
                    s["cross_disease_map"] = filtered_lags if filtered_lags else None
                    print(f"      {sid} ({target}): {s['cross_disease_map']}")

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
    # Step 3: Statistical Testing for Paired Scenarios
    # ------------------------------------------------------------------
    if not args.skip_scenarios:
        # Define base -> cross pairs
        pairs = [
            ("scenario1", "scenario2"),
            ("scenario1", "scenario7"),
            ("scenario3", "scenario4"),
            ("scenario3", "scenario8"),
            ("scenario5", "scenario6"),
            ("scenario5", "scenario9"),
        ]
        print(f"\n{'-'*60}")
        print("  STEP 3/3: Paired Wilcoxon Tests")
        print(f"{'-'*60}")
        for base_id, cross_id in pairs:
            # check if both exist in results dir
            if (Path(RESULTS_DIR) / base_id).exists() and (Path(RESULTS_DIR) / cross_id).exists():
                try:
                    cross_scenario_wilcoxon(base_id, cross_id, RESULTS_DIR)
                except Exception as e:
                    print(f"  >> [FAIL] Wilcoxon {base_id} vs {cross_id}: {e}")

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
