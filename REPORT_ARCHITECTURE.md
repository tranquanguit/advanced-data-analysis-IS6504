# REPORT ARCHITECTURE — Infectious Disease Forecasting (Climate-driven, Multi-province)

## Abstract (demo)
We present an empirical study comparing three forecasting paradigms for monthly infectious disease incidence across 53 Vietnamese provinces (1997–2017): (i) naïve/seasonal baselines and Prophet, (ii) tabular machine learning (XGBoost, HistGradientBoosting) with lag/rolling features, and (iii) deep sequence models (LSTM). Using a strict time split (train≤2014, val 2015, test 2016–2017) and multi-horizon targets (1–3 months), tabular models reduced MAE@1 by ~40% versus Naïve and ~30–35% versus Seasonal Naïve, while LSTM—despite extended tuning—did not surpass tree models. Climate lag analysis (rainfall, humidity) and cross-disease correlations (dengue, influenza, diarrhoea) reveal meaningful lagged relationships; SHAP highlights lagged disease and climate variables as dominant predictors. Results suggest that for moderate-length provincial time series, feature-engineered tabular models remain strong baselines, and negative outcomes for deep models should be reported transparently.

## Outline
1. **Introduction** (problem, motivation, climate impacts, contributions)
2. **Related Work** (classical TS, Prophet/SARIMA, tree models, deep seq models, climate–disease links)
3. **Data & Preprocessing**
   - Sources, provinces, period (1997–2017)
   - Variables: diseases (3), climate, socio
   - Cleaning: drop cols, ffill/bfill, scaling, splits
4. **Methodology**
   - Problem formulation & horizons (1/2/3 months)
   - Feature engineering (lags, rolling, seasonality encoding, optional cross-disease features)
   - Models: Naïve, SeasonalNaïve, Prophet, XGB/HGB, LSTM
   - Evaluation metrics (MAE, RMSE, SMAPE, outbreak P/R@95), significance tests
   - Scenarios matrix (targets, flags per SCENARIOS.md)
5. **Evaluation** (detailed structure)
   - 5.1 **Experimental Setup**
     - Time split (train/val/test), tuning protocol (val for grids), hardware note
     - Scenarios run (list IDs S1–S5; note flags used)
   - 5.2 **Overall Performance**
     - Table: MAE@1/2/3, RMSE, SMAPE for all models per main scenario (S1)
     - Outbreak metrics (precision/recall@95) for top 3 models
     - Significance vs SeasonalNaïve (Wilcoxon/Friedman+Nemenyi)
   - 5.3 **Model-type Comparison**
     - Highlight tabular vs Prophet vs LSTM; delta vs Naïve/Seasonal
     - Discuss negative result of LSTM (data regime, sequence length, tuning tried)
   - 5.4 **Ablations / Scenarios**
     - If run: effects of cross-disease features (S2) — delta MAE/PR
     - If run: per100k normalization (S3) — stability across provinces
     - Per-disease runs (S4/S5) — brief metric table, note differences
   - 5.5 **Per-Province Analysis**
     - MAE distribution across provinces (province_metrics.csv)
     - Identify best/worst provinces; possible reasons (data size, variability)
   - 5.6 **Explainability & Insights**
     - SHAP global top features; note lagged disease/climate dominance
     - SHAP by province: heterogeneity summary
     - PDP/ALE (if available) for top climate lags
   - 5.7 **EDA Findings**
     - Seasonality plots; climate–target lag heatmap (lag_correlation_heatmap)
     - Cross-disease corr & cross-corr: key r values, leading/lagging pairs
6. **Discussion**
   - Why tabular wins here; when LSTM might shine (longer series, richer exogenous signals)
   - Practical implications for surveillance (simple, reproducible models suffice)
   - Limitations: data length, reporting delays, unmodeled interventions
7. **Conclusion & Future Work**
   - Summarize gains; note negative deep-learning result
   - Future: hierarchical models, probabilistic forecasts, causal tests
8. **Appendix**
   - Full hyperparams, grids tried, additional scenario tables, extended plots
