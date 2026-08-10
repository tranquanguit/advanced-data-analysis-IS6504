"""Rolling-origin cross-validation across all forecasting scenarios.

Runs the existing 5-fold expanding-window CV (run_cv.run_cv_pipeline) for every
scenario defined in run_hybrid.SCENARIOS, applying the same dynamic NL-guided
configuration (from outputs/nonlinear/tables/suggested_lags.json) as the main
hybrid pipeline. Results for each scenario are saved separately under
results_cv/<scenario_id>/ so no run overwrites another.

Folds (expanding window, from run_cv.py):
    Test 2014: Train<=2012, Val 2013
    Test 2015: Train<=2013, Val 2014
    Test 2016: Train<=2014, Val 2015
    Test 2017: Train<=2015, Val 2016
    Test 2018: Train<=2016, Val 2017   (2018 is partial: Jan-Jun)

Usage:
    python run_cv_all.py                      # all scenarios except S7 (==S1)
    python run_cv_all.py --scenarios 1 2      # only S1, S2
    python run_cv_all.py --skip 7             # skip list (default: 7)
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
import yaml

from run_cv import run_cv_pipeline
from run_hybrid import SCENARIOS, load_suggested_lags


def _prevent_sleep() -> None:
    """Keep Windows awake while CV runs (prevents idle sleep killing long jobs)."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        print("[INFO] Sleep prevention enabled (SetThreadExecutionState).")
    except Exception as e:  # non-Windows or no permission
        print(f"[WARN] Could not enable sleep prevention: {e}")

BASE_CONFIG = "configs/default.yaml"
CV_RESULTS_DIR = Path("results_cv")
SUGGESTED_LAGS = Path("outputs/nonlinear/tables/suggested_lags.json")


def build_scenarios() -> list[dict]:
    """Replicate run_hybrid's scenario list incl. dynamic NL-guided override."""
    scs = copy.deepcopy(SCENARIOS)
    sug = load_suggested_lags(SUGGESTED_LAGS)
    if sug:
        for s in scs:
            if s["id"] in ("scenario7", "scenario8", "scenario9"):
                lags_for_target = sug.get(s["target"], {})
                filtered = {p: l for p, l in lags_for_target.items() if l}
                s["cross_disease_map"] = filtered if filtered else None
    return scs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rolling-origin CV across all scenarios")
    ap.add_argument("--scenarios", nargs="+", type=int, default=None,
                    help="Scenario numbers to run (default: all)")
    ap.add_argument("--skip", nargs="+", type=int, default=[7],
                    help="Scenario numbers to skip (default: 7, identical to S1)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _prevent_sleep()
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    scs = build_scenarios()
    if args.scenarios:
        scs = [s for s in scs if int(s["id"].replace("scenario", "")) in args.scenarios]
    skip = set(args.skip or [])

    CV_RESULTS_DIR.mkdir(exist_ok=True)
    summary_rows: list[dict] = []
    t_start = time.time()

    for s in scs:
        sid = s["id"]
        num = int(sid.replace("scenario", ""))
        if num in skip:
            print(f"[SKIP] {sid} (skip list)")
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["target"] = s["target"]
        cfg["experiment"]["cases_col"] = s["cases_col"]
        cfg["experiment"]["compute_rate_per100k"] = s["compute_rate_per100k"]
        cfg["experiment"]["cross_disease_map"] = s["cross_disease_map"]

        out_dir = CV_RESULTS_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["output_dir"] = str(out_dir).replace("\\", "/")

        tmp_cfg = out_dir / "_cv_config.yaml"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

        print(f"\n{'='*64}")
        print(f"  CV {sid}: target={s['target']}  cross_disease={s['cross_disease_map']}")
        print(f"{'='*64}")
        t0 = time.time()
        run_cv_pipeline(str(tmp_cfg))
        dt = time.time() - t0
        print(f"[OK] {sid} finished in {dt:.0f}s")

        # Aggregate mean +/- std across folds
        det_path = out_dir / "metrics" / "cv_results_detailed.csv"
        det = pd.read_csv(det_path)
        for model, g in det.groupby("model"):
            row = {"scenario": sid, "target": s["target"], "model": model,
                   "n_folds": int(len(g))}
            for h in range(1, 7):
                col = f"MAE@{h}"
                if col in g.columns:
                    row[f"{col}_mean"] = round(float(g[col].mean()), 4)
                    row[f"{col}_std"] = round(float(g[col].std()), 4)
            for m in ("precision", "recall"):
                if m in g.columns:
                    row[f"{m}_mean"] = round(float(g[m].mean()), 4)
            summary_rows.append(row)

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(CV_RESULTS_DIR / "cv_summary_all.csv", index=False)
        print(f"\n{'='*64}\n  CV SUMMARY (mean across folds) -> results_cv/cv_summary_all.csv")
        print(sdf[["scenario", "model", "MAE@1_mean", "MAE@1_std",
                   "MAE@3_mean", "MAE@6_mean"]].to_string(index=False))
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
