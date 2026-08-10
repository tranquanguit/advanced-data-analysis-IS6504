from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(
    df: pd.DataFrame,
    target: str,
    climate_cols: list[str],
    disease_cols: list[str] | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    month_profile = df.groupby("month")[target].mean()
    plt.figure(figsize=(8, 4))
    month_profile.plot(marker="o")
    plt.title("Average monthly dengue rate")
    plt.xlabel("Month")
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(out_dir / "seasonality_month_profile.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df[target], bins=30, kde=True)
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png")
    plt.close()

    corr_rows = []
    for c in climate_cols:
        if c not in df.columns:
            continue
        for lag in [0, 1, 2, 3, 4, 5, 6]:
            shifted = df.groupby("province")[c].shift(lag)
            corr = df[target].corr(shifted)
            corr_rows.append({"feature": c, "lag": lag, "corr": corr})

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "lag_correlation.csv", index=False)

    if not corr_df.empty:
        pivot = corr_df.pivot(index="feature", columns="lag", values="corr")
        plt.figure(figsize=(10, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Climate vs target lag correlation")
        plt.tight_layout()
        plt.savefig(out_dir / "lag_correlation_heatmap.png")
        plt.close()

    # Cross-disease correlation (optional)
    if disease_cols:
        disease_cols = [c for c in disease_cols if c in df.columns]
        if len(disease_cols) >= 2:
            corr_mat = df[disease_cols].corr()
            corr_mat.to_csv(out_dir / "disease_corr_matrix.csv")
            plt.figure(figsize=(4, 3))
            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Disease correlation (same month)")
            plt.tight_layout()
            plt.savefig(out_dir / "disease_corr_heatmap.png")
            plt.close()

            # Cross-correlation with lags 0-6 months
            xcorr_rows = []
            for d1 in disease_cols:
                for d2 in disease_cols:
                    if d1 == d2:
                        continue
                    for lag in [0, 1, 2, 3, 4, 5, 6]:
                        shifted = df.groupby("province")[d2].shift(lag)
                        corr = df[d1].corr(shifted)
                        xcorr_rows.append({"d1": d1, "d2": d2, "lag": lag, "corr": corr})
            xcorr_df = pd.DataFrame(xcorr_rows)
            xcorr_df.to_csv(out_dir / "disease_crosscorr.csv", index=False)
            if not xcorr_df.empty:
                pivot = xcorr_df.pivot_table(index=["d1", "d2"], columns="lag", values="corr")
                plt.figure(figsize=(8, 4))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
                plt.title("Disease cross-correlation by lag")
                plt.tight_layout()
                plt.savefig(out_dir / "disease_crosscorr_heatmap.png")
                plt.close()
