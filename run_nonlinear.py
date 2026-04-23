from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.nonlinear_analyzer import (
    analyze_global_dependencies,
    analyze_province_for_top_relationships,
    build_params,
    rank_relationships,
)
from src.data_loader import build_quality_reports, load_all_provinces_raw
from src.nonlinear_reporting import write_insights, write_markdown_summary, write_quality_summary
from src.runtime_config import load_runtime_config
from src.nonlinear_visualization import plot_heatmaps_by_target, plot_heatmaps_by_target_mi, plot_province_variability, plot_top_relationships


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-linear correlation analysis pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "plots": output_dir / "plots",
        "insights": output_dir / "insights",
        "reports": output_dir / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def dataset_stats(df: pd.DataFrame, province_col: str) -> dict:
    return {
        "rows": int(len(df)),
        "provinces": int(df[province_col].nunique()),
        "start_date": str(pd.to_datetime(df["date"]).min().date()),
        "end_date": str(pd.to_datetime(df["date"]).max().date()),
    }


def run_pipeline(config_path: str) -> None:
    cfg = load_runtime_config(config_path)
    analysis_cfg = cfg.analysis
    params = build_params(analysis_cfg)

    data_folder = Path(cfg.paths.get("data_folder", "data/raw"))
    output_dir = Path(cfg.paths.get("nonlinear_output_dir", "outputs/nonlinear"))
    out_dirs = ensure_output_dirs(output_dir)

    print(f"[INFO] Loading data from: {data_folder}")
    df = load_all_provinces_raw(data_folder, province_col=params.province_col)
    stats = dataset_stats(df, params.province_col)
    print(
        "[INFO] Dataset loaded: "
        f"rows={stats['rows']}, provinces={stats['provinces']}, "
        f"span={stats['start_date']} to {stats['end_date']}"
    )

    analysis_cols = [
        *params.disease_vars,
        *params.climate_vars,
        *params.social_vars,
    ]
    quality_reports = build_quality_reports(df, params.province_col, analysis_cols)
    quality_reports["missing_by_col"].to_csv(out_dirs["tables"] / "quality_missing_by_col.csv", index=False)
    quality_reports["missing_by_province"].to_csv(out_dirs["tables"] / "quality_missing_by_province.csv", index=False)
    quality_reports["static_by_province_col"].to_csv(out_dirs["tables"] / "quality_static_by_province_col.csv", index=False)

    print("[INFO] Computing global non-linear lagged dependencies...")
    global_df = analyze_global_dependencies(df, params)
    if global_df.empty:
        raise RuntimeError("No relationships were computed. Check config columns and thresholds.")
    global_df.to_csv(out_dirs["tables"] / "global_lag_metrics.csv", index=False)

    ranked_df = rank_relationships(global_df, analysis_cfg.get("ranking_weights", {}))
    ranked_df.to_csv(out_dirs["tables"] / "global_lag_metrics_ranked.csv", index=False)

    top_k = int(analysis_cfg.get("top_k_relationships", 30))
    top_df = ranked_df.head(top_k).copy()
    top_df.to_csv(out_dirs["tables"] / "top_relationships.csv", index=False)

    print("[INFO] Computing province-level heterogeneity for top relationships...")
    province_df = analyze_province_for_top_relationships(df, top_df, params)
    province_df.to_csv(out_dirs["tables"] / "province_top_relationships.csv", index=False)

    print("[INFO] Generating plots...")
    plot_heatmaps_by_target(ranked_df, out_dirs["plots"], metric_col="distance_corr")
    plot_heatmaps_by_target_mi(ranked_df, out_dirs["plots"], metric_col="mutual_info")
    plot_top_relationships(top_df, out_dirs["plots"] / "top_relationships.png", top_n=min(15, top_k))
    plot_province_variability(province_df, out_dirs["plots"] / "province_variability.png")

    print("[INFO] Writing insights and markdown summary...")
    write_insights(top_df, province_df, out_dirs["insights"] / "insights.txt", top_n=min(10, top_k))
    write_markdown_summary(
        stats,
        ranked_df,
        top_df,
        province_df,
        out_dirs["reports"] / "analysis_summary.md",
    )
    write_quality_summary(quality_reports, out_dirs["reports"] / "data_quality_summary.md")

    print("[INFO] Exporting suggested cross-disease lags (dCor > 0.4)...")
    export_suggested_lags(global_df, params, out_dirs["tables"] / "suggested_lags.json")

    print(f"[DONE] Analysis completed. Outputs saved to: {output_dir}")


def export_suggested_lags(global_df: pd.DataFrame, params, output_path: Path):
    """Filter lags for other diseases where distance_corr > 0.4."""
    out_map = {}
    for target in params.disease_vars:
        target_map = {}
        # We only care about cross-disease predictors
        other_diseases = [d for d in params.disease_vars if d != target]
        
        for predictor in other_diseases:
            # Filter global_df for this pair
            mask = (global_df["target"] == target) & (global_df["predictor"] == predictor)
            matches = global_df[mask]
            
            # Sub-filter for dCor > 0.4
            significant = matches[matches["distance_corr"] > 0.4]
            lags = sorted(significant["lag"].unique().tolist())
            target_map[predictor] = [int(l) for l in lags]
            
        out_map[target] = target_map

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_map, f, indent=4)
    print(f"[INFO] Saved {len(out_map)} targets' suggested lags to {output_path.name}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
