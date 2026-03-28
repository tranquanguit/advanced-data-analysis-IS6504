# NEXT ACTIONS — Dengue Forecasting (updated 2026-03-28)

## 1) Data & Target (Owner: Data & EDA lead, Due: 2026-03-30)
- [ ] Drop non-informative cols (`Unnamed: 0`, `year_month`) before feature building; re-run preprocess.
- [ ] Normalize target as cases per 100k population; keep raw cases as secondary target for sensitivity.
- [ ] Standardize/scale continuous features per province (or z-score global) to help LSTM/XGB stability.
- [ ] Ensure train/test split is strictly time-based and identical across all models; add rolling-origin CV with 3 folds for robustness.
- [ ] Adopt 3-way time split: train ≤ 2014-12-31, val 2015-01-01 → 2015-12-31 (for tuning/early stopping), test 2016-01-01 → 2017-12-31; update config + splitter accordingly.

## 2) Baselines & Experiment Framing (Owner: Baseline & Experiment, Due: 2026-03-31)
- [ ] Lock baseline suite: Naive (t-1), Seasonal Naive (t-12), Prophet (per province), SARIMA quick check on 3 representative provinces.
- [ ] Define three experiment tracks (documented in report): Model-type (Naive/Prophet/XGB/HGB/LSTM), Input ablation (disease-only vs +climate vs +socio), Lag/window (lag sets 1,3,6; rolling window 3/6).
- [ ] Add pooled-global vs per-province comparison for XGB/HGB; keep metrics per horizon (1,2,3).

## 3) Tabular ML refinement (Owner: Tabular ML, Due: 2026-04-01)
- [ ] Refit XGB/HGB after data cleaning; tune `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` via 30-trial Optuna (MAE@1 focus).
- [ ] Calibrate outbreak classification at 90th/95th percentile; report precision/recall/F1; export calibrated thresholds.
- [ ] Re-run SHAP on best tabular model after column cleanup; drop population-size leakage by using rates features.

## 4) LSTM revamp (Owner: Deep Learning, Due: 2026-04-03)
- [ ] Build sequence loader (seq_len=12, stride=1), batch_size≈64, optimizer AdamW lr=1e-3, weight_decay=1e-4, dropout=0.2.
- [ ] Add validation split (last 12 months of train); early stopping on MAE@1; gradient clipping 1.0.
- [ ] Try separate head per horizon (three linear heads) and compare to shared head; report MAE@1/2/3 vs HGB.

## 5) Insight & Lag analysis (Owner: Data & EDA lead, Due: 2026-04-04)
- [ ] Compute CCF lag plots for top climate vars per region (North/Central/South); summarize optimal lags.
- [ ] Cluster provinces by climate + dengue pattern (k-means on monthly profiles); map SHAP importance by cluster.
- [ ] Generate PDP/ALE for top 5 SHAP features to evidence nonlinear effects; add 2-sentence takeaway per plot.

## 6) Statistics & Significance (Owner: Experiment, Due: 2026-04-05)
- [ ] Replace pairwise Wilcoxon with Friedman + Nemenyi across models for MAE@1; keep Wilcoxon vs SeasonalNaive as appendix.
- [ ] Bootstrap 1000x confidence intervals for MAE@1 per model; include in report table.

## 7) Reporting package (Owner: Report lead, Due: 2026-04-06)
- [ ] Update tables: model_comparison, province_metrics, outbreak metrics (90/95), ablation results, significance tests.
- [ ] Curate plots: seasonality by month, lag CCF heatmaps, prediction curves (top 2 models), SHAP summary, PDP/ALE panels.
- [ ] Draft Contribution section with three bullets: model-type comparison, climate/socio value-add, lagged nonlinear insight; align with supervisor feedback.
- [ ] Prepare 1-page executive summary highlighting numeric gains (target ≥20–40% better MAE@1 than Naive; ≥10–20% better than SeasonalNaive).

## 8) Stretch (optional, start after 2026-04-06)
- [ ] Granger causality on pooled and 5 key provinces to strengthen causal hint.
- [ ] Sensitivity analysis on train_end/test_start dates to show stability.
