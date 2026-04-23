from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# --- Workaround for SHAP + XGBoost >= 2.1.0 compatibility bug ---
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

def _extract_estimators(model) -> list:
    """Extract individual estimators from MultiOutputRegressor or return model as a list."""
    if hasattr(model, "estimators_"):
        return list(model.estimators_)
    return [model]

def run_shap_analysis(model, x_background, x_test, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimators = _extract_estimators(model)
    
    all_shap_values = []
    
    for i, est in enumerate(estimators):
        try:
            # TreeExplainer is preferred for trees (XGB, HistGB, LGBM)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(x_test)
        except Exception as e:
            print(f"[WARN] TreeExplainer failed for estimator {i} ({e}), falling back to generic Explainer...")
            explainer = shap.Explainer(est, x_background)
            sv = explainer(x_test).values
        
        # Ensure sv is a numpy array (TreeExplainer sometimes returns lists/etc weirdly)
        if isinstance(sv, list):
            sv = sv[0] if len(sv) > 0 else np.zeros_like(x_test)
        all_shap_values.append(sv)

    # Calculate global importance by averaging absolute SHAP across all horizons
    if all_shap_values:
        # all_shap_values is list of arrays [N, F]
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in all_shap_values], axis=0)
        main_shap_for_plot = all_shap_values[0] # Visualize the first horizon (t+1)
    else:
        importance = np.zeros(len(feature_names))
        main_shap_for_plot = np.zeros_like(x_test)

    # Plot for the primary horizon (t+1)
    plt.figure()
    shap.summary_plot(main_shap_for_plot, x_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()

    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    df.to_csv(output_dir / "top_features.csv", index=False)
    return df

def shap_by_province(model, df_test: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimators = _extract_estimators(model)
    
    # We explain the first estimator (t+1) for province-level breakdown to save computation
    est = estimators[0]
    rows = []
    
    try:
        explainer = shap.TreeExplainer(est)
    except Exception:
        print("[WARN] TreeExplainer not compatible for province-level breakdown.")
        return pd.DataFrame()

    for province, g in df_test.groupby("province"):
        if len(g) < 10:
            continue
        x_p = g[feature_cols]
        try:
            sv = explainer.shap_values(x_p)
            if isinstance(sv, list):
                sv = sv[0]
            mean_abs = np.abs(sv).mean(axis=0)
            
            for feature, imp in zip(feature_cols, mean_abs):
                rows.append({"province": province, "feature": feature, "importance": float(imp)})
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / "shap_by_province.csv", index=False)
    return out
