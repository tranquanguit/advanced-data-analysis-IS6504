from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(model, x_background, x_test, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, x_background)
    shap_values = explainer(x_test)

    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()

    importance = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    df.to_csv(output_dir / "top_features.csv", index=False)
    return df


def shap_by_province(model, df_test: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    explainer = shap.Explainer(model)

    for province, g in df_test.groupby("province"):
        if len(g) < 10:
            continue
        x_p = g[feature_cols]
        values = explainer(x_p)
        mean_abs = np.abs(values.values).mean(axis=0)
        for feature, imp in zip(feature_cols, mean_abs):
            rows.append({"province": province, "feature": feature, "importance": float(imp)})

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "shap_by_province.csv", index=False)
    return out
