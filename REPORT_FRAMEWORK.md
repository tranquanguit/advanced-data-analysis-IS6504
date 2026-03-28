# REPORT FRAMEWORK — What to keep & how to read results

## 1) Artifacts to keep for reporting
- Metrics:
  - `outputs/metrics/model_comparison.csv` (overall MAE/RMSE/SMAPE @1/2/3, outbreak precision/recall).
  - `outputs/metrics/province_metrics.csv` (MAE per province for best model).
  - `outputs/metrics/significance_vs_seasonal.csv` (Wilcoxon vs SeasonalNaive).
- Predictions:
  - `outputs/predictions/pred_<Model>.csv` (actual/pred per horizon for plotting/error analysis).
- Plots:
  - `outputs/plots/seasonality_month_profile.png`, `target_distribution.png` (EDA).
  - `outputs/plots/lag_correlation_heatmap.png` (climate lag effect).
  - `outputs/plots/prediction_<Model>.png` (top-2 models).
  - `outputs/plots/disease_corr_heatmap.png`, `outputs/plots/disease_crosscorr_heatmap.png` (correlation & cross-correlation giữa bệnh).
- SHAP / Insight (khi bật enable_shap):
  - `outputs/shap/shap_summary.png`, `top_features.csv`, `shap_by_province.csv`, `insights.txt`.

## 2) How to read & what to state in the report
- Performance:
  - Quote MAE@1 (primary) and MAE@2/3; compare to Naive & SeasonalNaive to show scientific meaning.
  - Outbreak precision/recall at 95th percentile (from model_comparison.csv).
  - Mention significance vs SeasonalNaive (p-value from significance_vs_seasonal.csv).
- Best model selection:
  - Pick model with lowest MAE@1; if close, prefer higher outbreak recall.
- Per-province reliability:
  - Use province_metrics.csv; highlight worst/best provinces and variance.
- EDA climate:
  - From lag_correlation_heatmap: cite top 1–2 climate variables and optimal lag range (e.g., rainfall lag 2).
- Cross-disease insight:
  - From disease_corr_heatmap / disease_crosscorr.csv: note any moderate correlation (|r|≈0.3–0.5) and leading/lagging disease pairs.
- SHAP (if enabled):
  - List top 3 features, note presence of lagged disease/climate features; mention regional heterogeneity if shap_by_province shows spread.

## 3) Minimal narrative structure
1. Problem & data split (train/val/test dates).
2. Models compared (Naive, Seasonal, Prophet, XGB/HGB, LSTM) + tuning note.
3. Main results (table: MAE@1/2/3, outbreak P/R); significance vs Seasonal.
4. Insights:
   - Climate lag effects (heatmap).
   - Cross-disease correlations (optional if relevant to scenario).
   - Feature importance (SHAP).
5. Limitations & next steps (e.g., try cross-disease features, per-100k normalization).
