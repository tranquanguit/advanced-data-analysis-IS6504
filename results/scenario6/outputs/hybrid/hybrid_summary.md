# Hybrid Report Summary

## Positioning
- Primary contribution: non-linear lagged relationship analysis.
- Secondary validation: forecasting performance against simple baselines.

## Forecasting Validation
- Primary metric: `MAE@1`
- Best model: `HistGB` with MAE@1=6.1890
- Improvement vs `Naive`: 20.77%
- Improvement vs `SeasonalNaive`: 36.88%

### Significance Snapshot
- Prophet: p_value=1.025e-20
- Naive: p_value=3.523e-05
- LSTM: p_value=0.006325
- HistGB: p_value=0.006413
- XGBoost: p_value=0.2773

## Non-Linear Findings
- Diarrhoea_rates <- Influenza_rates (lag=0) | score=0.996, distance_corr=0.513, spearman=0.538, mutual_info=0.347
- Influenza_rates <- Diarrhoea_rates (lag=0) | score=0.964, distance_corr=0.518, spearman=0.538, mutual_info=0.297
- Influenza_rates <- Diarrhoea_rates (lag=1) | score=0.849, distance_corr=0.471, spearman=0.467, mutual_info=0.256
- Influenza_rates <- Diarrhoea_rates (lag=2) | score=0.827, distance_corr=0.455, spearman=0.458, mutual_info=0.253
- Influenza_rates <- Diarrhoea_rates (lag=5) | score=0.807, distance_corr=0.435, spearman=0.448, mutual_info=0.257
- Influenza_rates <- Diarrhoea_rates (lag=4) | score=0.806, distance_corr=0.427, spearman=0.454, mutual_info=0.261
- Diarrhoea_rates <- Influenza_rates (lag=1) | score=0.805, distance_corr=0.443, spearman=0.446, mutual_info=0.247
- Diarrhoea_rates <- population_urban (lag=0) | score=0.800, distance_corr=0.415, spearman=-0.425, mutual_info=0.291
- Influenza_rates <- Diarrhoea_rates (lag=3) | score=0.797, distance_corr=0.450, spearman=0.463, mutual_info=0.213
- Diarrhoea_rates <- population_urban (lag=5) | score=0.778, distance_corr=0.378, spearman=-0.431, mutual_info=0.299

## Auto Insights
- Top non-linear lagged relationships:
- - Diarrhoea_rates <- Influenza_rates (lag=0) | score=0.996, spearman=0.538, distance_corr=0.513, mutual_info=0.347
- - Influenza_rates <- Diarrhoea_rates (lag=0) | score=0.964, spearman=0.538, distance_corr=0.518, mutual_info=0.297
- - Influenza_rates <- Diarrhoea_rates (lag=1) | score=0.849, spearman=0.467, distance_corr=0.471, mutual_info=0.256
- - Influenza_rates <- Diarrhoea_rates (lag=2) | score=0.827, spearman=0.458, distance_corr=0.455, mutual_info=0.253
- - Influenza_rates <- Diarrhoea_rates (lag=5) | score=0.807, spearman=0.448, distance_corr=0.435, mutual_info=0.257
- - Influenza_rates <- Diarrhoea_rates (lag=4) | score=0.806, spearman=0.454, distance_corr=0.427, mutual_info=0.261
- - Diarrhoea_rates <- Influenza_rates (lag=1) | score=0.805, spearman=0.446, distance_corr=0.443, mutual_info=0.247
- - Diarrhoea_rates <- population_urban (lag=0) | score=0.800, spearman=-0.425, distance_corr=0.415, mutual_info=0.291
- - Influenza_rates <- Diarrhoea_rates (lag=3) | score=0.797, spearman=0.463, distance_corr=0.450, mutual_info=0.213
- - Diarrhoea_rates <- population_urban (lag=5) | score=0.778, spearman=-0.431, distance_corr=0.378, mutual_info=0.299
- Strongest relationship per target disease:
- - Diarrhoea_rates: predictor=Influenza_rates, lag=0, distance_corr=0.513
- - Influenza_rates: predictor=Diarrhoea_rates, lag=0, distance_corr=0.518
- Top relationships with strongest province heterogeneity:
- - Influenza_rates <- population_urban (lag 4): std(distance_corr)=0.178
- - Influenza_rates <- population_urban (lag 1): std(distance_corr)=0.176
- - Diarrhoea_rates <- Influenza_rates (lag 5): std(distance_corr)=0.164
- - Diarrhoea_rates <- Influenza_rates (lag 6): std(distance_corr)=0.163
- - Diarrhoea_rates <- Influenza_rates (lag 4): std(distance_corr)=0.161

## Non-Linear Subproject Summary (Raw)

# Non-Linear Correlation Analysis Summary

## Dataset
- Rows: 14520
- Provinces: 55
- Time span: 1997-01-01 to 2018-12-01

## Coverage
- Global relationships tested: 378
- Top relationships retained: 30
- Province-level records for top relationships: 1650

## Key Takeaway
- Strongest relationship: `Diarrhoea_rates <- Influenza_rates (lag 0)`, composite score=0.996.

## Reporting Guidance
- Put non-linear findings as the main Results section.
- Use forecasting table only as validation that extracted signals are operationally useful.