from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated hybrid report from forecasting and non-linear outputs")
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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def pct_improvement(base: float, value: float) -> float:
    if base == 0:
        return 0.0
    return (base - value) / base * 100.0


def build_hybrid_summary(
    forecasting_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    nonlinear_top_df: pd.DataFrame,
    nonlinear_insights: str,
    nonlinear_summary_md: str,
    report_cfg: dict,
) -> str:
    metric = report_cfg.get("target_metric", "MAE@1")
    baseline_name = report_cfg.get("baseline_model", "Naive")
    seasonal_baseline_name = report_cfg.get("seasonal_baseline_model", "SeasonalNaive")
    top_n_nl = int(report_cfg.get("top_n_nonlinear", 10))

    lines = [
        "# Hybrid Report Summary",
        "",
        "## Positioning",
        "- Primary contribution: non-linear lagged relationship analysis.",
        "- Secondary validation: forecasting performance against simple baselines.",
        "",
    ]

    if forecasting_df.empty:
        lines.extend(
            [
                "## Forecasting Validation",
                "- Forecasting metrics file not found.",
                "",
            ]
        )
    else:
        lines.append("## Forecasting Validation")
        lines.append(f"- Primary metric: `{metric}`")
        ranked = forecasting_df.sort_values(metric).reset_index(drop=True)
        best = ranked.iloc[0]
        lines.append(f"- Best model: `{best['model']}` with {metric}={best[metric]:.4f}")

        m = forecasting_df.set_index("model")
        if baseline_name in m.index:
            base_val = float(m.loc[baseline_name, metric])
            imp = pct_improvement(base_val, float(best[metric]))
            lines.append(f"- Improvement vs `{baseline_name}`: {imp:.2f}%")
        if seasonal_baseline_name in m.index:
            seas_val = float(m.loc[seasonal_baseline_name, metric])
            imp = pct_improvement(seas_val, float(best[metric]))
            lines.append(f"- Improvement vs `{seasonal_baseline_name}`: {imp:.2f}%")
        lines.append("")

        if not significance_df.empty:
            lines.append("### Significance Snapshot")
            for _, row in significance_df.sort_values("p_value").head(5).iterrows():
                lines.append(f"- {row['model']}: p_value={row['p_value']:.4g}")
            lines.append("")

    if nonlinear_top_df.empty:
        lines.extend(
            [
                "## Non-Linear Findings",
                "- Non-linear top relationships file not found.",
                "",
            ]
        )
    else:
        lines.append("## Non-Linear Findings")
        for _, row in nonlinear_top_df.head(top_n_nl).iterrows():
            lines.append(
                "- "
                f"{row['target']} <- {row['predictor']} (lag={int(row['lag'])}) | "
                f"score={row['composite_score']:.3f}, "
                f"distance_corr={row['distance_corr']:.3f}, "
                f"spearman={row['spearman_corr']:.3f}, "
                f"mutual_info={row['mutual_info']:.3f}"
            )
        lines.append("")

    if nonlinear_insights.strip():
        lines.append("## Auto Insights")
        lines.extend([f"- {ln}" for ln in nonlinear_insights.splitlines() if ln.strip()])
        lines.append("")

    if nonlinear_summary_md.strip():
        lines.append("## Non-Linear Subproject Summary (Raw)")
        lines.append("")
        lines.append(nonlinear_summary_md.strip())
        lines.append("")

    lines.append("## Reporting Guidance")
    lines.append("- Put non-linear findings as the main Results section.")
    lines.append("- Use forecasting table only as validation that extracted signals are operationally useful.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    report_cfg = cfg.get("report", {})

    output_dir = resolve_path(config_dir, paths_cfg.get("output_dir", "../outputs/hybrid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    forecasting_metrics_path = resolve_path(
        config_dir, paths_cfg.get("forecasting_metrics_file", "../outputs/metrics/model_comparison.csv")
    )
    forecasting_significance_path = resolve_path(
        config_dir, paths_cfg.get("forecasting_significance_file", "../outputs/metrics/significance_vs_seasonal.csv")
    )
    nonlinear_top_path = resolve_path(
        config_dir, paths_cfg.get("nonlinear_top_relationships_file", "../non-linear-correlation-analysis/outputs/tables/top_relationships.csv")
    )
    nonlinear_insights_path = resolve_path(
        config_dir, paths_cfg.get("nonlinear_insights_file", "../non-linear-correlation-analysis/outputs/insights/insights.txt")
    )
    nonlinear_summary_path = resolve_path(
        config_dir, paths_cfg.get("nonlinear_summary_file", "../non-linear-correlation-analysis/outputs/reports/analysis_summary.md")
    )

    forecasting_df = read_csv_if_exists(forecasting_metrics_path)
    significance_df = read_csv_if_exists(forecasting_significance_path)
    nonlinear_top_df = read_csv_if_exists(nonlinear_top_path)
    nonlinear_insights = read_text_if_exists(nonlinear_insights_path)
    nonlinear_summary_md = read_text_if_exists(nonlinear_summary_path)

    if not forecasting_df.empty:
        forecasting_df.to_csv(tables_dir / "forecasting_model_comparison.csv", index=False)
    if not significance_df.empty:
        significance_df.to_csv(tables_dir / "forecasting_significance.csv", index=False)
    if not nonlinear_top_df.empty:
        top_n = int(report_cfg.get("top_n_nonlinear", 10))
        nonlinear_top_df.head(top_n).to_csv(tables_dir / "nonlinear_top_relationships.csv", index=False)

    summary = build_hybrid_summary(
        forecasting_df=forecasting_df,
        significance_df=significance_df,
        nonlinear_top_df=nonlinear_top_df,
        nonlinear_insights=nonlinear_insights,
        nonlinear_summary_md=nonlinear_summary_md,
        report_cfg=report_cfg,
    )

    summary_path = output_dir / "hybrid_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[DONE] Hybrid summary written to: {summary_path}")


if __name__ == "__main__":
    main()

