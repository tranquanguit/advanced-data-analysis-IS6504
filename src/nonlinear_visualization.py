from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmaps_by_target(global_df: pd.DataFrame, output_dir: Path, metric_col: str = "distance_corr") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if global_df.empty:
        return

    for target, g in global_df.groupby("target"):
        pivot = g.pivot_table(index="predictor", columns="lag", values=metric_col, aggfunc="mean")
        if pivot.empty:
            continue
        plt.figure(figsize=(10, max(4, 0.5 * len(pivot))))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title(f"{metric_col} by lag: target={target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{target}_{metric_col}.png", dpi=150)
        plt.close()

def plot_heatmaps_by_target_mi(global_df: pd.DataFrame, output_dir: Path, metric_col: str = "mutual_info") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if global_df.empty:
        return

    for target, g in global_df.groupby("target"):
        pivot = g.pivot_table(index="predictor", columns="lag", values=metric_col, aggfunc="mean")
        if pivot.empty:
            continue
        plt.figure(figsize=(10, max(4, 0.5 * len(pivot))))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title(f"{metric_col} by lag: target={target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{target}_{metric_col}.png", dpi=150)
        plt.close()


def plot_top_relationships(top_df: pd.DataFrame, output_file: Path, top_n: int = 15) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if top_df.empty:
        return

    view = top_df.head(top_n).copy()
    view["label"] = view.apply(lambda r: f"{r['target']} <- {r['predictor']} (lag {int(r['lag'])})", axis=1)
    view = view.iloc[::-1]

    plt.figure(figsize=(12, max(5, 0.4 * len(view))))
    plt.barh(view["label"], view["composite_score"])
    plt.xlabel("Composite score")
    plt.title(f"Top {top_n} non-linear relationships")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_province_variability(province_df: pd.DataFrame, output_file: Path, top_n_relationships: int = 8) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if province_df.empty:
        return

    top_rel = province_df["relationship"].drop_duplicates().head(top_n_relationships).tolist()
    view = province_df[province_df["relationship"].isin(top_rel)].copy()
    if view.empty:
        return

    plt.figure(figsize=(12, max(5, 0.5 * len(top_rel))))
    sns.boxplot(data=view, y="relationship", x="distance_corr", orient="h")
    plt.title("Province variability of distance correlation (top relationships)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
