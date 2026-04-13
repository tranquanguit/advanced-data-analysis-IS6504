# Hybrid Report Summary

## Positioning
- Primary contribution: non-linear lagged relationship analysis.
- Secondary validation: forecasting performance against simple baselines.

## Forecasting Validation
- Primary metric: `MAE@1`
- Best model: `HistGB` with MAE@1=6.8993
- Improvement vs `Naive`: 11.68%
- Improvement vs `SeasonalNaive`: 29.63%

### Significance Snapshot
- Prophet: p_value=1.549e-17
- Naive: p_value=3.523e-05
- LSTM: p_value=0.00237
- XGBoost: p_value=0.7534
- HistGB: p_value=0.7888

## Non-Linear Findings
- Influenza_rates <- Diarrhoea_rates (lag=0) | score=0.795, distance_corr=0.552, spearman=0.538, mutual_info=0.309
- Diarrhoea_rates <- Influenza_rates (lag=0) | score=0.793, distance_corr=0.543, spearman=0.538, mutual_info=0.343
- Diarrhoea_rates <- toilet_rate (lag=1) | score=0.756, distance_corr=0.530, spearman=-0.506, mutual_info=0.308
- Diarrhoea_rates <- toilet_rate (lag=5) | score=0.750, distance_corr=0.529, spearman=-0.500, mutual_info=0.300
- Diarrhoea_rates <- toilet_rate (lag=3) | score=0.745, distance_corr=0.519, spearman=-0.502, mutual_info=0.303
- Diarrhoea_rates <- toilet_rate (lag=6) | score=0.740, distance_corr=0.516, spearman=-0.501, mutual_info=0.288
- Diarrhoea_rates <- toilet_rate (lag=2) | score=0.734, distance_corr=0.501, spearman=-0.504, mutual_info=0.319
- Diarrhoea_rates <- toilet_rate (lag=0) | score=0.729, distance_corr=0.497, spearman=-0.503, mutual_info=0.314
- Diarrhoea_rates <- toilet_rate (lag=4) | score=0.718, distance_corr=0.487, spearman=-0.500, mutual_info=0.294
- Influenza_rates <- Diarrhoea_rates (lag=3) | score=0.698, distance_corr=0.503, spearman=0.463, mutual_info=0.245

## Auto Insights
- Top non-linear lagged relationships:
- - Influenza_rates <- Diarrhoea_rates (lag=0) | score=0.795, spearman=0.538, distance_corr=0.552, mutual_info=0.309
- - Diarrhoea_rates <- Influenza_rates (lag=0) | score=0.793, spearman=0.538, distance_corr=0.543, mutual_info=0.343
- - Diarrhoea_rates <- toilet_rate (lag=1) | score=0.756, spearman=-0.506, distance_corr=0.530, mutual_info=0.308
- - Diarrhoea_rates <- toilet_rate (lag=5) | score=0.750, spearman=-0.500, distance_corr=0.529, mutual_info=0.300
- - Diarrhoea_rates <- toilet_rate (lag=3) | score=0.745, spearman=-0.502, distance_corr=0.519, mutual_info=0.303
- - Diarrhoea_rates <- toilet_rate (lag=6) | score=0.740, spearman=-0.501, distance_corr=0.516, mutual_info=0.288
- - Diarrhoea_rates <- toilet_rate (lag=2) | score=0.734, spearman=-0.504, distance_corr=0.501, mutual_info=0.319
- - Diarrhoea_rates <- toilet_rate (lag=0) | score=0.729, spearman=-0.503, distance_corr=0.497, mutual_info=0.314
- - Diarrhoea_rates <- toilet_rate (lag=4) | score=0.718, spearman=-0.500, distance_corr=0.487, mutual_info=0.294
- - Influenza_rates <- Diarrhoea_rates (lag=3) | score=0.698, spearman=0.463, distance_corr=0.503, mutual_info=0.245
- Strongest relationship per target disease:
- - Influenza_rates: predictor=Diarrhoea_rates, lag=0, distance_corr=0.552
- - Diarrhoea_rates: predictor=Influenza_rates, lag=0, distance_corr=0.543
- Top relationships with strongest province heterogeneity:
- - Influenza_rates <- toilet_rate (lag 6): std(distance_corr)=0.205
- - Influenza_rates <- toilet_rate (lag 2): std(distance_corr)=0.205
- - Influenza_rates <- Diarrhoea_rates (lag 3): std(distance_corr)=0.158
- - Influenza_rates <- Diarrhoea_rates (lag 1): std(distance_corr)=0.157
- - Influenza_rates <- Diarrhoea_rates (lag 0): std(distance_corr)=0.157

## Non-Linear Subproject Summary (Raw)

# Non-Linear Correlation Analysis Summary

## Dataset
- Rows: 14520
- Provinces: 55
- Time span: 1997-01-01 to 2018-12-01

## Coverage
- Global relationships tested: 294
- Top relationships retained: 30
- Province-level records for top relationships: 1601

## Key Takeaway
- Strongest relationship: `Influenza_rates <- Diarrhoea_rates (lag 0)`, composite score=0.795.

## Reporting Guidance
- Put non-linear findings as the main Results section.
- Use forecasting table only as validation that extracted signals are operationally useful.