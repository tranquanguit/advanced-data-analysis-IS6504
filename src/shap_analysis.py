from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# --- Workaround for SHAP + XGBoost >= 2.1.0 compatibility bug ---
# SHAP <= 0.45 fails because XGBoost >= 2.1.0 outputs base_score as an array (e.g., '[0.5]')
# instead of a string float. We monkey-patch the ubjson decoder in SHAP to fix this.
try:
    import shap.explainers._tree
    _old_decode = shap.explainers._tree.decode_ubjson_buffer

    def _custom_decode(fd):
        jmodel = _old_decode(fd)
        try:
            val = jmodel.get("learner", {}).get("learner_model_param", {}).get("base_score")
            if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                import json
                jmodel["learner"]["learner_model_param"]["base_score"] = str(json.loads(val)[0])
        except Exception:
            pass
        return jmodel

    shap.explainers._tree.decode_ubjson_buffer = _custom_decode
except ImportError:
    pass
# -------------------------------------------------------------


def run_shap_analysis(model, x_background, x_test, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    # TreeExplainer is more stable across XGBoost versions than the generic Explainer path.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()

    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    df.to_csv(output_dir / "top_features.csv", index=False)
    return df


def shap_by_province(model, df_test: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    explainer = shap.TreeExplainer(model)

    for province, g in df_test.groupby("province"):
        if len(g) < 10:
            continue
        x_p = g[feature_cols]
        values = explainer.shap_values(x_p)
        mean_abs = np.abs(values).mean(axis=0)
        for feature, imp in zip(feature_cols, mean_abs):
            rows.append({"province": province, "feature": feature, "importance": float(imp)})

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "shap_by_province.csv", index=False)
    return out
