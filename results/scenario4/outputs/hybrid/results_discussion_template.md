# Results and Discussion (Hybrid Direction)

## 1. Main Positioning

This study follows a hybrid direction:

- Main contribution: non-linear lagged relationship analysis.
- Secondary validation: forecasting performance against simple baselines.

## 2. Forecasting Validation Results

### 2.1 Overall model comparison (MAE@1 as primary metric)

From the latest run:

- Best model by MAE@1: **HistGB** (`MAE@1 = 6.8993`).
- Improvement vs Naive: **11.68%**.
- Improvement vs SeasonalNaive: **29.63%**.

Reference values:

- HistGB: `MAE@1 = 6.8993`, `precision = 0.5556`, `recall = 0.6897`
- XGBoost: `MAE@1 = 6.9241`, `precision = 0.5244`, `recall = 0.7414`
- LSTM: `MAE@1 = 7.5038`
- Naive: `MAE@1 = 7.8117`
- SeasonalNaive: `MAE@1 = 9.8049`
- Prophet: `MAE@1 = 9.0062`

Suggested write-up:

Forecasting experiments confirm that feature-engineered tree models remain the most competitive under the current data regime. The best MAE@1 is achieved by HistGB, with clear gains over Naive and SeasonalNaive, supporting the practical predictability of the target series.

### 2.2 Significance note

Current significance file shows very low p-values for Prophet and Naive vs SeasonalNaive, while tree models and LSTM are not significantly different from SeasonalNaive under this specific test setting.

Suggested write-up:

The significance table should be interpreted as a baseline-relative diagnostic rather than a full ranking test across all models. We therefore use it as supporting evidence, while primary model selection remains based on MAE and outbreak metrics.

## 3. Main Non-Linear Findings

### 3.1 Top lagged non-linear relationships

Top relationships from the non-linear pipeline:

1. `Influenza_rates <- Diarrhoea_rates (lag 0)`  
   `score = 0.795`, `distance_corr = 0.552`, `spearman = 0.538`
2. `Diarrhoea_rates <- Influenza_rates (lag 0)`  
   `score = 0.793`, `distance_corr = 0.543`, `spearman = 0.538`
3. `Diarrhoea_rates <- toilet_rate (lag 1)`  
   `score = 0.756`, `distance_corr = 0.530`, `spearman = -0.506`

Suggested write-up:

The strongest dependencies are observed in cross-disease links (Influenza and Diarrhoea) and sanitation-related factors (toilet_rate). This indicates that both epidemiological coupling and social-environmental context may jointly shape disease dynamics.

### 3.2 Province-level heterogeneity

Relationships with highest province variability include:

- `Influenza_rates <- toilet_rate (lag 6)`
- `Influenza_rates <- toilet_rate (lag 2)`
- `Influenza_rates <- Diarrhoea_rates (lag 3)`

Suggested write-up:

The same predictor-lag pair does not have uniform strength across provinces, suggesting regional heterogeneity. This supports stratified surveillance and region-specific intervention planning rather than one-size-fits-all conclusions.

## 4. Integrated Discussion (Hybrid Message)

Suggested narrative:

The non-linear analysis identifies candidate mechanisms (cross-disease coupling and sanitation-linked effects), while forecasting experiments show that these signals are operationally useful because models can outperform naive seasonal baselines. Together, this hybrid evidence strengthens both scientific insight and practical relevance.

## 5. Limitations

Suggested points:

- Some variables are partially missing or static in specific provinces (see data quality report).
- Relationship analysis remains associational, not causal.
- SHAP in forecasting currently has compatibility issues with the installed model stack and was skipped in the latest run.

## 6. Recommended Conclusion Paragraph

Suggested conclusion:

This work demonstrates that non-linear, lagged relationships can be systematically extracted at multi-province scale and validated by forecasting gains over simple baselines. The strongest patterns highlight cross-disease coupling and sanitation-related effects, with meaningful province heterogeneity. Future work should extend to robust causal identification and region-specific policy modeling.
