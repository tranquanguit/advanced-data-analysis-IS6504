# Multi-Disease Multi-Horizon Infectious-Disease Forecasting for Vietnam

Research code, data, and results accompanying the paper
*"Do Cross-Disease Signals Improve Multi-Horizon Infectious-Disease Forecasting
Across Vietnam?"*.

The pipeline studies monthly incidence of **dengue fever**, **influenza**, and
**diarrhoea** across **55 Vietnamese provinces** (1998–2018), together with 12
climate variables and 4 socio-demographic indicators. It has two parts:

1. **Non-linear dependence analysis** — Distance Correlation, Mutual Information,
   Spearman, Kendall, and Pearson under seasonal/regional confound control, to
   rank lagged relationships between climate, social, and disease variables.
2. **Direct multi-horizon forecasting** — a 12-month input window and a 6-month
   horizon (one model per horizon), comparing Naive, Seasonal Naive, Prophet,
   XGBoost, HistGradientBoosting, LightGBM, and LSTM, with SHAP explainability and
   a rolling-origin cross-validation.

## Repository structure

```
src/            Core library (data loading, features, models, evaluation, NL analysis, SHAP)
configs/        YAML experiment configuration
data/           raw/ (per-province Excel) and processed/ (modelling table)
results/        Per-scenario forecasting outputs (S1–S9): metrics, predictions, plots, SHAP
results_cv/     Rolling-origin cross-validation results
outputs/        Latest-run artifacts (non-linear tables/plots, EDA, predictions)
run_all.py      Single-scenario forecasting pipeline
run_nonlinear.py    Non-linear dependence analysis only
run_hybrid.py   Orchestrator: non-linear analysis + all scenarios + tests
run_cv.py / run_cv_all.py    Rolling-origin cross-validation
aggregate_cv.py Aggregate CV results (mean ± std)
```

## Quick start

```bash
pip install -r requirements.txt
python run_hybrid.py            # full pipeline (analysis + scenarios)
python run_cv_all.py            # rolling-origin cross-validation
```

All parameters (targets, horizons, model grids, paths) are in
`configs/default.yaml`.

## Notes

- The Seasonal Naive / Naive baseline alignment, the outbreak-threshold source,
  and the significance test are being revised for a leakage-free re-run; see the
  code comments in `src/models/naive.py` and `src/evaluation.py`.
- Data are monthly and aggregated at the provincial level, with no individually
  identifiable information.

## License / data

Research/educational use. Disease surveillance and meteorological data are
provided by the respective Vietnamese national institutes; please confirm the
data-use terms before redistribution.
