from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid analysis: forecasting + non-linear + integrated report")
    parser.add_argument("--config", default="configs/hybrid_report.yaml", help="Path to hybrid YAML config")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(config_dir: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    root_dir = Path(__file__).resolve().parent
    cfg = load_config(config_path)

    paths_cfg = cfg.get("paths", {})
    run_cfg = cfg.get("run", {})
    continue_on_step_error = bool(run_cfg.get("continue_on_step_error", True))

    output_dir = resolve_path(config_dir, paths_cfg.get("output_dir", "../outputs/hybrid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "hybrid_run.log"

    with run_log.open("a", encoding="utf-8") as log:
        log.write(f"\n=== Hybrid run started at {datetime.now().isoformat()} ===\n")
        log.write(f"Config: {config_path}\n")

    step_errors: list[str] = []

    if run_cfg.get("run_forecasting", True):
        forecasting_config = resolve_path(config_dir, paths_cfg.get("forecasting_config", "../configs/default.yaml"))
        try:
            run_command([sys.executable, "run_all.py", "--config", str(forecasting_config)], cwd=root_dir)
        except subprocess.CalledProcessError as exc:
            msg = f"Forecasting step failed: {exc}"
            step_errors.append(msg)
            print(f"[WARN] {msg}")
            if not continue_on_step_error:
                raise

    if run_cfg.get("run_nonlinear", True):
        nonlinear_project_dir = resolve_path(
            config_dir, paths_cfg.get("nonlinear_project_dir", "../non-linear-correlation-analysis")
        )
        nonlinear_config_raw = paths_cfg.get("nonlinear_config", "configs/default.yaml")
        nonlinear_config_path = Path(nonlinear_config_raw)
        if not nonlinear_config_path.is_absolute():
            candidate = (nonlinear_project_dir / nonlinear_config_path).resolve()
            if candidate.exists():
                nonlinear_config_path = candidate
            else:
                nonlinear_config_path = resolve_path(config_dir, nonlinear_config_raw)
        try:
            run_command(
                [sys.executable, "run_analysis.py", "--config", str(nonlinear_config_path)],
                cwd=nonlinear_project_dir,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"Non-linear step failed: {exc}"
            step_errors.append(msg)
            print(f"[WARN] {msg}")
            if not continue_on_step_error:
                raise

    if run_cfg.get("build_report", True):
        try:
            run_command([sys.executable, "build_hybrid_report.py", "--config", str(config_path)], cwd=root_dir)
        except subprocess.CalledProcessError as exc:
            msg = f"Report build step failed: {exc}"
            step_errors.append(msg)
            print(f"[WARN] {msg}")
            if not continue_on_step_error:
                raise

    with run_log.open("a", encoding="utf-8") as log:
        log.write(f"Completed at {datetime.now().isoformat()}\n")
        for err in step_errors:
            log.write(f"ERROR: {err}\n")
    if step_errors:
        print(f"[DONE] Hybrid pipeline completed with warnings. Output: {output_dir}")
    else:
        print(f"[DONE] Hybrid pipeline completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
