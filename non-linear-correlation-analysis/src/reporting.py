from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_insights(top_df: pd.DataFrame, province_df: pd.DataFrame, output_file: Path, top_n: int = 10) -> list[str]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    if top_df.empty:
        lines.append("No valid relationships were found with current thresholds.")
        output_file.write_text("\n".join(lines), encoding="utf-8")
        return lines

    lines.append("Top non-linear lagged relationships:")
    for _, row in top_df.head(top_n).iterrows():
        lines.append(
            f"- {row['target']} <- {row['predictor']} (lag={int(row['lag'])}) | "
            f"score={row['composite_score']:.3f}, "
            f"spearman={row['spearman_corr']:.3f}, "
            f"distance_corr={row['distance_corr']:.3f}, "
            f"mutual_info={row['mutual_info']:.3f}"
        )

    strongest_by_target = (
        top_df.sort_values("composite_score", ascending=False).groupby("target", as_index=False).head(1)
    )
    lines.append("")
    lines.append("Strongest relationship per target disease:")
    for _, row in strongest_by_target.iterrows():
        lines.append(
            f"- {row['target']}: predictor={row['predictor']}, lag={int(row['lag'])}, "
            f"distance_corr={row['distance_corr']:.3f}"
        )

    if not province_df.empty:
        heterogeneity = (
            province_df.groupby("relationship")["distance_corr"].std().sort_values(ascending=False).head(5)
        )
        lines.append("")
        lines.append("Top relationships with strongest province heterogeneity:")
        for rel, std_val in heterogeneity.items():
            lines.append(f"- {rel}: std(distance_corr)={std_val:.3f}")

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return lines


def write_markdown_summary(
    dataset_stats: dict,
    global_df: pd.DataFrame,
    top_df: pd.DataFrame,
    province_df: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Non-Linear Correlation Analysis Summary",
        "",
        "## Dataset",
        f"- Rows: {dataset_stats.get('rows', 'NA')}",
        f"- Provinces: {dataset_stats.get('provinces', 'NA')}",
        f"- Time span: {dataset_stats.get('start_date', 'NA')} to {dataset_stats.get('end_date', 'NA')}",
        "",
        "## Coverage",
        f"- Global relationships tested: {len(global_df)}",
        f"- Top relationships retained: {len(top_df)}",
        f"- Province-level records for top relationships: {len(province_df)}",
        "",
        "## Key Takeaway",
    ]

    if top_df.empty:
        lines.append("- No relationships passed the current thresholds.")
    else:
        best = top_df.iloc[0]
        lines.append(
            "- Strongest relationship: "
            f"`{best['target']} <- {best['predictor']} (lag {int(best['lag'])})`, "
            f"composite score={best['composite_score']:.3f}."
        )

    output_file.write_text("\n".join(lines), encoding="utf-8")


def write_quality_summary(quality_reports: dict[str, pd.DataFrame], output_file: Path, top_n: int = 10) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Summary", ""]

    missing_by_col = quality_reports.get("missing_by_col", pd.DataFrame())
    missing_by_province = quality_reports.get("missing_by_province", pd.DataFrame())
    static_df = quality_reports.get("static_by_province_col", pd.DataFrame())

    lines.append("## Highest Missing Ratio Columns")
    if missing_by_col.empty:
        lines.append("- No data.")
    else:
        for _, row in missing_by_col.head(top_n).iterrows():
            lines.append(f"- {row['column']}: missing_ratio={row['missing_ratio']:.3f}")

    lines.append("")
    lines.append("## Provinces With Highest Average Missing Ratio")
    if missing_by_province.empty:
        lines.append("- No data.")
    else:
        province_col = missing_by_province.columns[0]
        for _, row in missing_by_province.head(top_n).iterrows():
            lines.append(f"- {row[province_col]}: avg_missing_ratio={row['avg_missing_ratio']:.3f}")

    lines.append("")
    lines.append("## Potentially Static Pairs (n_unique <= 2)")
    if static_df.empty:
        lines.append("- No data.")
    else:
        province_col = static_df.columns[0]
        view = static_df[static_df["n_unique"] <= 2].head(top_n)
        if view.empty:
            lines.append("- None detected in top rows.")
        else:
            for _, row in view.iterrows():
                lines.append(
                    f"- {row[province_col]} / {row['column']}: n_unique={int(row['n_unique'])}, std={row['std']:.6f}"
                )

    output_file.write_text("\n".join(lines), encoding="utf-8")


def write_hybrid_bridge_note(
    top_df: pd.DataFrame,
    forecasting_metrics_df: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Hybrid Bridge Note", ""]

    if top_df.empty:
        lines.append("- No top non-linear relationships found.")
    else:
        best = top_df.iloc[0]
        lines.append(
            "- Main non-linear finding: "
            f"`{best['target']} <- {best['predictor']} (lag {int(best['lag'])})` "
            f"with composite score {best['composite_score']:.3f}."
        )

    if forecasting_metrics_df.empty:
        lines.append("- Forecasting metrics not available.")
    else:
        m = forecasting_metrics_df.set_index("model")
        if "Naive" in m.index:
            best_model = forecasting_metrics_df.sort_values("MAE@1").iloc[0]
            best_name = best_model["model"]
            naive = float(m.loc["Naive", "MAE@1"])
            best_mae = float(best_model["MAE@1"])
            improvement = (naive - best_mae) / naive * 100 if naive else 0.0
            lines.append(
                "- Forecasting validation: "
                f"best model `{best_name}` improves MAE@1 by {improvement:.2f}% versus Naive."
            )

    output_file.write_text("\n".join(lines), encoding="utf-8")
