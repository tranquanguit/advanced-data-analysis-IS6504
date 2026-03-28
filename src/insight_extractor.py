from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_insights(top_features_path: Path, shap_by_province_path: Path, output_path: Path) -> list[str]:
    top_df = pd.read_csv(top_features_path)
    prov_df = pd.read_csv(shap_by_province_path)

    insights: list[str] = []
    top3 = top_df.head(3)["feature"].tolist()
    insights.append(f"Top 3 important features globally: {top3}")

    climate = [f for f in top3 if any(k in f.lower() for k in ["rain", "humidity", "temp"]) ]
    if climate:
        insights.append(f"Climate variables contribute strongly: {climate}")

    lag_feats = [f for f in top3 if "lag" in f.lower()]
    if lag_feats:
        insights.append(f"Lag effects detected from top features: {lag_feats}")

    if not prov_df.empty:
        regional_var = prov_df.groupby("feature")["importance"].std().sort_values(ascending=False)
        if len(regional_var) > 0:
            insights.append(f"Strongest regional heterogeneity observed on: {regional_var.index[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(insights), encoding="utf-8")
    return insights
