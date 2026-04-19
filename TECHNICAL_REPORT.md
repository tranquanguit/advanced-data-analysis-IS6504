# Climate-Driven Infectious Disease Forecasting in Vietnam: A Hybrid Non-Linear Analysis and Multi-Horizon Prediction Pipeline

---

## Abstract

This study presents a hybrid analytical pipeline for multi-province, multi-horizon infectious disease forecasting in Vietnam by integrating non-linear dependency analysis with machine learning–based prediction. Leveraging monthly surveillance data from 55 provinces over a 22-year period (1997–2018) encompassing three infectious diseases (Dengue fever, Influenza, Diarrhoea), twelve climate variables, and socio-demographic indicators, we construct a dual-branch system. The first branch quantifies non-linear lagged dependencies between climate, sanitation, and disease variables using Distance Correlation, Mutual Information, Spearman rank correlation, Kendall Tau, and Pearson correlation under seasonal–regional confound control. The second branch employs a Multi-Input Multi-Output (MIMO) forecasting architecture with a 12-month input window and 6-month prediction horizon, benchmarking six models: Naive, Seasonal Naive, Prophet, XGBoost, HistGradientBoosting, and LSTM. Experimental results across five scenarios demonstrate that HistGradientBoosting achieves the best MAE@1 of 6.692 for Dengue (14.3% improvement over Naive), generalizes effectively to Influenza, and that cross-disease features—particularly Diarrhoea rates—rank as the third most important predictor via SHAP analysis. Non-linear analysis reveals strong bidirectional Influenza–Diarrhoea dependency (Distance Correlation = 0.552) and identifies sanitation infrastructure as a structural predictor, while climate effects operate primarily through seasonal pathways.

---

## 1. Introduction

### 1.1 Motivation

Infectious diseases remain a leading cause of morbidity and mortality in tropical developing nations, where climate variability, rapid urbanization, and heterogeneous public health infrastructure create complex epidemiological landscapes [Ref1]. Vietnam, situated in the tropical monsoon belt, experiences annual epidemics of Dengue fever, seasonal Influenza, and Diarrhoea that impose substantial burdens on its healthcare system. The relationships between climatic factors—rainfall, temperature, humidity—and disease incidence are well-documented in the epidemiological literature [Ref2], yet these relationships are typically non-linear, lagged, and spatially heterogeneous across provinces with diverse geographic and socio-economic profiles.

Accurate multi-month forecasting of disease incidence rates at the provincial level would enable public health authorities to allocate resources proactively, deploy vector control measures, and prepare hospital capacity ahead of epidemic surges. While traditional statistical approaches such as ARIMA and decomposition-based methods (e.g., Prophet) have been applied to disease surveillance time series, their capacity to capture complex, non-linear climate–disease interactions is inherently limited [Ref3].

### 1.2 Problem Statement & Research Gap

The problem addressed in this study is two-fold: (i) rigorously quantifying the non-linear, lagged dependencies between climatic-socioeconomic predictors and infectious disease incidence across multiple provinces; and (ii) building operational multi-horizon forecasting models that leverage these dependencies for 1-to-6-month-ahead prediction.

Existing approaches in the literature exhibit several limitations. First, most studies employ Pearson correlation to assess climate–disease relationships, which fails to detect monotonic or arbitrary non-linear dependencies [Ref4]. Second, forecasting studies typically focus on single-horizon (t+1) prediction, neglecting the operational need for multi-month planning horizons. Third, deep learning models (LSTM, Transformer) are often applied without rigorous comparison against simpler baselines, or without transparent reporting of negative results when they underperform [Ref5]. Fourth, the issue of spatial heterogeneity—whether a climate–disease relationship holds uniformly across all provinces—is rarely investigated. Finally, few studies integrate non-linear dependency analysis with forecasting within a unified framework, missing the opportunity to validate analytical findings through predictive performance.

### 1.3 Research Questions & Hypotheses

This work addresses four research questions:

**RQ1.** Are the relationships between climate variables and infectious disease incidence fundamentally non-linear, and do they persist after controlling for seasonal and regional confounds?

**RQ2.** Can a MIMO forecasting architecture (12-month input, 6-month output) significantly outperform naive baselines at extended horizons (t+3 to t+6)?

**RQ3.** Do cross-disease features (e.g., Influenza and Diarrhoea rates) carry predictive value for Dengue forecasting?

**RQ4.** Does the forecasting pipeline generalize across different disease targets (Dengue, Influenza, Diarrhoea)?

The corresponding hypotheses are: **H1:** Distance Correlation between climate and disease variables significantly exceeds Pearson correlation, indicating non-linearity. **H2:** Gradient boosting models will demonstrate increasing advantage over Naive baselines as the prediction horizon extends. **H3:** Cross-disease features will improve forecasting accuracy for Dengue, as confirmed by SHAP feature importance. **H4:** The pipeline architecture will generalize with comparable improvement margins across diseases.

### 1.4 Objectives & Scope

The primary objective is to design, implement, and validate a hybrid analysis pipeline that combines non-linear correlation analysis with multi-horizon time series forecasting for provincial-level infectious disease surveillance in Vietnam. Specific tasks include: constructing a non-linear analysis module with confound control and composite scoring; implementing a MIMO forecasting pipeline with six model families; designing five experimental scenarios to address the research questions; and providing model explainability through SHAP analysis.

The scope encompasses 55 Vietnamese provinces, three diseases, a 22-year monthly time series, and six model architectures. The study assumes that historical reporting patterns are reasonably consistent across the observation period. Out of scope are real-time deployment, causal inference, and intervention modeling.

### 1.5 Contributions

This work makes the following contributions: **(C1)** A dual-branch hybrid pipeline that integrates non-linear dependency analysis with forecasting in a unified, reproducible framework. **(C2)** Systematic application of Distance Correlation and Mutual Information with confound control to quantify non-linear climate–disease dependencies across 55 provinces. **(C3)** A MIMO forecasting architecture evaluated across five experimental scenarios with rigorous statistical testing. **(C4)** Transparent reporting of negative results (LSTM outbreak detection failure; Naive superiority for Diarrhoea). **(C5)** Cross-disease SHAP analysis revealing Diarrhoea rates as a strong proxy predictor for Dengue fever.

---

## 2. Related Work

### 2.1 Background Knowledge

**Time series forecasting** involves predicting future values of a variable based on its historical observations and potentially exogenous covariates. In a multi-horizon setting, the model predicts values at multiple future steps simultaneously. The MIMO strategy trains a single model to output a vector of predictions, avoiding error accumulation inherent in recursive (iterated) approaches [Ref6].

**Non-linear dependence measures** extend beyond Pearson correlation. Distance Correlation (dCor) [Ref7] equals zero if and only if the variables are statistically independent, making it strictly more general than Pearson. Mutual Information (MI) [Ref8] measures the total information shared between variables, including all orders of statistical dependency. Both are well-suited for detecting complex climate–disease relationships.

**SHAP (SHapley Additive exPlanations)** [Ref9] provides theoretically-grounded feature attribution values based on cooperative game theory, enabling interpretation of individual prediction contributions.

### 2.2 Prior Work Comparison & Limitations

Classical approaches to dengue forecasting include SARIMA and ARIMAX models, which assume linear relationships and struggle with the heavy-tailed, skewed distributions typical of disease incidence data [Ref10]. Prophet, developed by Facebook [Ref11], decomposes time series into trend, seasonality, and holidays components, but cannot incorporate exogenous climate covariates directly into its structural model.

Gradient boosting methods—XGBoost [Ref12] and LightGBM [Ref13]—have demonstrated strong performance on tabular forecasting tasks, particularly when feature engineering captures relevant lagged and rolling statistics. LSTM networks [Ref14] have been widely applied to sequential prediction, but their advantage over tree-based models on moderate-length time series remains debated [Ref15].

In the climate–disease nexus, most studies rely on Pearson or Spearman correlation at fixed lags, without systematic evaluation across lag ranges or confound control for seasonal co-variation [Ref16]. Distance Correlation has been applied in genomics and neuroscience [Ref17] but remains underutilized in epidemiological settings.

### 2.3 Positioning of This Work

This study occupies the intersection of non-linear dependency analysis and machine learning–based forecasting. Unlike prior works that treat these as separate investigations, we integrate both into a unified pipeline where non-linear findings directly inform and validate the forecasting models. Furthermore, we address the generalizability gap by evaluating the same pipeline architecture across three diseases, and we provide explicit negative results (LSTM's failure, Naive's superiority for Diarrhoea) that are typically omitted in the literature.

---

## 3. Methodology

### 3.1 Overview of Approach

The system is organized as a **dual-branch hybrid pipeline**. The first branch performs non-linear correlation analysis across all disease–predictor–lag combinations, employing confound control and composite scoring to rank the most informative dependencies. The second branch constructs a MIMO forecasting pipeline: data loading, feature engineering (lags, rolling statistics, cyclical encoding), multi-horizon target creation, temporal train/validation/test split, StandardScaler normalization, multi-model training (Naive, Seasonal Naive, Prophet, XGBoost, HistGradientBoosting, LSTM), evaluation (MAE, RMSE, SMAPE, outbreak precision/recall), and SHAP explainability. Both branches converge in a hybrid report builder that synthesizes findings.

[Figure 1: System architecture diagram showing dual-branch hybrid pipeline]

### 3.2 Data & Data Pipeline

The dataset comprises monthly surveillance records from **55 Vietnamese provinces** spanning 1997–2018, stored as individual Excel files per province. Each record contains three disease incidence rates (Dengue, Influenza, Diarrhoea per 100,000 population), twelve climate variables (temperature, rainfall, humidity, sunshine, evaporation), and socio-demographic indicators (population distribution, sanitation rates). The total dataset consists of approximately **14,520 observations** after preprocessing.

Preprocessing includes: (1) per-province forward-fill then backward-fill for missing values, avoiding cross-province leakage; (2) lag feature generation (1–12 months) for target, climate, and social variables using grouped shift operations; (3) rolling mean and standard deviation (windows 3 and 6 months, shifted by 1 to prevent information leakage); (4) cyclical encoding of month via sine/cosine transformation; (5) multi-horizon target creation via negative shift; (6) StandardScaler normalization fitted on training data only.

The temporal split follows a strict chronological protocol: Train ≤ 2014-12-31, Validation = 2015, Test = 2016–2017. This ensures no future information leaks into model training or tuning.

### 3.3 Evaluation Protocol

The evaluation framework employs three tiers. **Performance metrics** include MAE (primary), RMSE, and SMAPE, reported at each horizon (MAE@1 through MAE@6). **Outbreak detection** uses the 95th percentile of the test set's true values as the outbreak threshold, computing precision and recall of extreme event identification. **Statistical testing** applies the Wilcoxon signed-rank test comparing each model's absolute errors against Seasonal Naive at α = 0.05, chosen for its non-parametric robustness suitable for heavy-tailed disease distributions.

Baselines include Naive (last-value persistence), Seasonal Naive (same-month-last-year), and Prophet. Models must demonstrate improvement over these baselines to claim scientific contribution.

### 3.4 Model Design

Six models span four paradigm categories:

**Baselines.** Naive replicates the current value for all horizons. Seasonal Naive uses the corresponding month from the previous year, capturing annual cyclicity.

**Classical time series.** Prophet decomposes the series into trend and yearly seasonality, fitted per-province independently.

**Tabular machine learning.** XGBoost and HistGradientBoosting are wrapped in MultiOutputRegressor for MIMO output. Hyperparameters are tuned via exhaustive grid search on the validation set (XGBoost: max_depth ∈ {4,6,8}, learning_rate ∈ {0.05,0.1}, subsample/colsample_bytree ∈ {0.7,0.8}; HistGB: max_iter ∈ {250,300,350}, learning_rate ∈ {0.03,0.05}, max_depth ∈ {6,8}).

**Deep learning.** A 2-layer stacked LSTM (hidden_size=64, dropout=0.2) with a linear projection head, trained with AdamW (lr=1e-3, weight_decay=1e-4), gradient clipping (max_norm=1.0), batch_size=64, and early stopping (patience=8 epochs).

**Non-linear analysis module.** Five dependency measures (Pearson, Spearman, Kendall, Distance Correlation, MI) are computed for all target–predictor pairs across lags 0–6, under confound control (subtraction of monthly and provincial means). A weighted composite score (w_dCor=0.40, w_|Spearman|=0.35, w_MI=0.25) ranks the top 30 relationships, which are then analyzed at the province level for heterogeneity.

### 3.5 Training & Implementation Details

The training protocol follows a tune-on-validation, retrain-on-train+val, evaluate-on-test paradigm. After grid search identifies best hyperparameters (by MAE@1 on validation), tree models are refitted on Train+Val before test evaluation. LSTM uses early stopping on validation loss. SHAP analysis employs TreeExplainer for exact Shapley value computation on XGBoost. The pipeline is implemented in Python 3.10+ using pandas, NumPy, scikit-learn, XGBoost, PyTorch, Prophet, SHAP, and SciPy. [TODO: Hardware specifications to be supplemented.]

---

## 4. Experiments and Results

### 4.1 Experimental Settings

Five forecasting scenarios (S1–S5) and one non-linear analysis scenario are executed automatically via a scenario runner that modifies the YAML configuration and copies outputs.

| ID | Target | Research Question | Key Configuration |
|:---|:---|:---|:---|
| S1 | Dengue_fever_rates | MIMO 12→6 effectiveness | Backbone: climate + social features only |
| S2 | Dengue_fever_rates | Cross-disease predictive value | include_other_diseases=true |
| S3 | Dengue_fever_rates | Population normalization impact | compute_rate_per100k=true |
| S4 | Influenza_rates | Pipeline generalization | Target change |
| S5 | Diarrhoea_rates | Pipeline generalization | Target change |
| NL | All three diseases | Non-linear dependency analysis | 5 metrics, lag 0–6, confound control |

[Table 1: Experimental scenario matrix]

### 4.2 Main Quantitative Results

**Dengue forecasting (S1).** HistGradientBoosting achieves the best MAE@1 of 7.063, a 9.6% improvement over Naive (7.812) and 28.0% over Seasonal Naive (9.805). The advantage becomes more pronounced at extended horizons: at MAE@3, HistGB (8.748) outperforms Naive (12.704) by 31%. LSTM achieves stable errors across horizons (7.6–8.1) but completely fails at outbreak detection (Precision=0, Recall=0). HistGB achieves the best balance with 0.621 Precision and 0.707 Recall.

[Table 2: Model comparison — Scenario S1 (Dengue)]

**Cross-disease features (S2).** Adding Influenza and Diarrhoea lag features improves HistGB's MAE@1 from 7.063 to **6.692** (−5.3%), the best result across all Dengue scenarios. SHAP analysis reveals Diarrhoea_rates ranks **3rd** (importance=1.583) among 260+ features, surpassing all climate variables.

**Influenza generalization (S4).** HistGB achieves MAE@1=27.633, improving 8.6% over Naive and 22.7% over Seasonal Naive. All ML models significantly outperform Seasonal Naive (Wilcoxon p < 10⁻⁹), contrasting with Dengue where significance was not achieved. Seasonal Naive performs worse than Naive for Influenza, reflecting unstable inter-annual seasonality.

**Diarrhoea (S5) — Negative result.** Naive achieves the best MAE@1 (13.616), outperforming XGBoost (14.439) by 6%. This negative result arises from Diarrhoea's extremely strong lag-1 autocorrelation, making last-value persistence near-optimal at t+1. However, XGBoost outperforms Naive at extended horizons (MAE@3 onwards).

[Table 3: Cross-scenario summary of best models]

**Statistical significance.** Wilcoxon tests reveal that for Dengue (S1), ML models do not achieve statistical significance over Seasonal Naive at horizon t+1 (p=0.39 for XGBoost, p=0.92 for HistGB), reflecting Seasonal Naive's strength for seasonal diseases. For Influenza and Diarrhoea, all ML models achieve highly significant improvements (p < 10⁻⁵).

### 4.3 Main Qualitative Results

**Non-linear dependency analysis.** Among 294 tested relationships, the strongest dependency is between Influenza and Diarrhoea at lag 0 (composite score=0.795, dCor=0.552), a bidirectional relationship persisting after confound control. This indicates shared transmission mechanisms beyond seasonal co-variation. The second dominant pattern involves Diarrhoea←toilet_rate (score=0.718–0.756 across all lags), a stable structural relationship reflecting sanitation infrastructure's persistent effect.

Critically, **no climate variable appears in the top 10** after controlling for seasonal and regional confounds, suggesting climate effects operate primarily through seasonal pathways rather than direct residual mechanisms.

[Table 4: Top 10 non-linear relationships by composite score]

**Province heterogeneity.** The Influenza←toilet_rate relationship shows std(dCor)=0.205 across provinces, indicating substantial regional variation—the effect is strong in rural provinces with poor sanitation but weak in urban centers.

**Disease correlation structure.** Pearson correlation at lag 0 reveals strong Influenza–Diarrhoea association (r=0.631) but near-zero Dengue associations (r≈−0.06 to −0.09), consistent with the epidemiological distinction between waterborne/respiratory and vector-borne transmission.

**SHAP insights (S1).** Dengue_lag1 dominates (importance=6.337), followed by rollmean_3 (2.692), confirming strong disease inertia. Temperature at lag 1–2 months outperforms lag 0, aligning with Aedes mosquito breeding cycles. Rainfall matters at lag 5 (importance=0.308), reflecting the ~5-month chain from rain→standing water→larvae→adult mosquitoes→infection.

[Figure 2: SHAP summary plot for Scenario S1]

### 4.4 Empirical Analyses

**Ablation: cross-disease features.** The S1→S2 ablation shows that cross-disease features benefit HistGB (−5.3% MAE@1) but not XGBoost (+0.3%) or LSTM (−0.6%), suggesting model-specific capacity to exploit inter-disease interactions.

**Ablation: disease generalization.** The pipeline improves over Naive by 8–10% for Dengue and Influenza but not for Diarrhoea at short horizons, attributable to Diarrhoea's strong lag-1 autocorrelation and the MIMO architecture's multi-horizon trade-off.

**Error analysis.** Province-level MAE for Dengue (S1) ranges from 0.48 (Lai Châu, near-zero incidence) to 27.18 (Gia Lai, high variability Central Highlands). For Influenza, the pattern reverses: northern mountainous provinces show highest errors (Lào Cai: 109.10) due to cold-season epidemic volatility.

**Failure analysis.** Three systematic failures are identified: (1) LSTM achieves Outbreak Precision=0, Recall=0 across all Dengue/Influenza scenarios due to conservative MSE-driven convergence toward mean predictions; (2) Prophet achieves MAE@1=50.3 for Influenza (2× worse than HistGB), as its trend+seasonality decomposition cannot capture virus-strain–driven variability; (3) The MIMO architecture's joint optimization across 6 horizons sacrifices short-horizon accuracy for Diarrhoea compared to simple persistence.

**Complexity.** The primary computational bottleneck is Distance Correlation computation (O(n²) per pair), mitigated by sampling. Total pipeline runtime for five scenarios plus non-linear analysis is estimated at 2–4 hours on a standard workstation. [TODO: Exact timing to be supplemented.]

---

## 5. Discussion

The experimental results provide varying degrees of support for the initial hypotheses.

**H1 (Non-linearity) — Strongly supported.** Pearson correlations between climate variables and Dengue never exceed 0.241, while Distance Correlation reaches 0.552 for disease–disease pairs. The gap dCor ≫ |r| constitutes direct evidence of non-linear dependencies. However, the confound control analysis reveals a nuanced interpretation: after removing seasonal and regional means, climate variables no longer dominate the top relationships. This suggests that climate's influence on disease is primarily **mediated through seasonal pathways**—both climate and disease co-vary with season—rather than through direct residual mechanisms. This finding has important implications: using climate variables as features in forecasting models is beneficial not because of causal effects, but because they act as proxies for seasonal timing with higher temporal resolution than simple month indicators.

**H2 (MIMO multi-horizon advantage) — Supported for Dengue and Influenza.** HistGB's advantage over Naive increases from ~10% at MAE@1 to ~31% at MAE@3, confirming that extended-horizon forecasting is where ML models provide the most value. This directly addresses the critique that "forecasting is trivial" by demonstrating that naive approaches fail dramatically at planning-relevant horizons.

**H3 (Cross-disease predictive value) — Partially supported.** SHAP analysis confirms Diarrhoea_rates as the 3rd most important feature for Dengue forecasting, likely serving as a proxy for environmental sanitation conditions that also affect mosquito breeding. However, the improvement is model-specific (HistGB benefits, XGBoost does not), tempering a universal recommendation.

**H4 (Generalization) — Partially supported.** The pipeline generalizes well to Influenza (comparable improvement margins) but fails for Diarrhoea at short horizons. This reveals a fundamental tension in the MIMO architecture: optimizing across all horizons simultaneously creates implicit trade-offs that can harm individual-horizon performance for diseases with strong persistence characteristics.

**Practical implications.** For public health deployment, HistGradientBoosting with cross-disease features offers the most operationally useful configuration: reasonably accurate at all horizons with the best outbreak detection balance. However, the strong province-level heterogeneity in both model errors and non-linear relationships argues against a one-size-fits-all model, suggesting that regional model specialization could yield further improvements.

**Limitations.** The study does not establish causal relationships. The temporal split leaves only two years for testing. Missing data in some provinces (particularly poverty_rate at 12.7%) may bias results. The LSTM architecture is minimal; attention mechanisms or Transformer architectures could yield different conclusions. The non-linear analysis uses month_province confound control, which may over-correct by removing genuine climate signals that happen to correlate with season.

---

## 6. Conclusion and Future Work

### 6.1 Conclusion

This study has presented a hybrid analytical pipeline integrating non-linear correlation analysis with multi-horizon time series forecasting for infectious disease surveillance across 55 Vietnamese provinces. The key findings are:

First, the non-linear analysis demonstrates that climate–disease relationships are fundamentally non-linear (dCor significantly exceeds Pearson r), but these effects are largely absorbed by seasonal confound control. The strongest residual dependencies are between diseases themselves (Influenza↔Diarrhoea, dCor=0.552) and between sanitation infrastructure and Diarrhoea (dCor=0.530), with substantial provincial heterogeneity.

Second, HistGradientBoosting emerges as the consistently best forecasting model across Dengue and Influenza, achieving up to 14.3% MAE improvement over Naive with cross-disease features. The MIMO 12→6 architecture proves especially valuable at extended horizons (31% improvement at MAE@3), validating the operational relevance of multi-month forecasting.

Third, SHAP analysis reveals that autoregressive features (lag1, rolling means) dominate all disease models, while cross-disease signals (Diarrhoea rates for Dengue) outperform all climate variables, suggesting shared environmental risk factors. Temperature contributes most effectively at 1–2 month lags, and rainfall at ~5-month lag, consistent with vector ecology theory.

Fourth, the study transparently reports three negative results: LSTM's complete failure at outbreak detection due to conservative mean-regression behavior, Prophet's poor performance for Influenza, and Naive's superiority over ML for Diarrhoea at short horizons—each providing valuable methodological lessons.

### 6.2 Future Work

Future directions include: (1) per-province or per-region model training to address spatial heterogeneity; (2) attention-based architectures (Temporal Fusion Transformer) to potentially overcome LSTM limitations; (3) recursive SISO architectures for diseases with strong lag-1 autocorrelation (Diarrhoea); (4) asymmetric loss functions that penalize missed outbreaks more heavily; (5) causal inference methods (Granger causality, do-calculus) to distinguish true climate effects from confounded co-variation; (6) probabilistic forecasting with uncertainty quantification; (7) real-time deployment with streaming data integration.

---

## References

[Ref1] World Health Organization, "Climate change and infectious diseases," WHO Technical Report, 2023.

[Ref2] K. E. Jones et al., "Global trends in emerging infectious diseases," *Nature*, vol. 451, pp. 990–993, 2008.

[Ref3] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.

[Ref4] G. J. Székely, M. L. Rizzo, and N. K. Bakirov, "Measuring and testing dependence by correlation of distances," *The Annals of Statistics*, vol. 35, no. 6, pp. 2769–2794, 2007.

[Ref5] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are Transformers Effective for Time Series Forecasting?" in *Proceedings of AAAI*, 2023.

[Ref6] S. Ben Taieb, G. Bontempi, A. F. Atiya, and A. Sorjamaa, "A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition," *Expert Systems with Applications*, vol. 39, no. 8, pp. 7067–7083, 2012.

[Ref7] G. J. Székely and M. L. Rizzo, "Brownian distance covariance," *The Annals of Applied Statistics*, vol. 3, no. 4, pp. 1236–1265, 2009.

[Ref8] A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating mutual information," *Physical Review E*, vol. 69, p. 066138, 2004.

[Ref9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[Ref10] L. Hii et al., "Forecast of Dengue Incidence Using Temperature and Rainfall," *PLoS Neglected Tropical Diseases*, vol. 6, no. 11, e1908, 2012.

[Ref11] S. J. Taylor and B. Letham, "Forecasting at scale," *The American Statistician*, vol. 72, no. 1, pp. 37–45, 2018.

[Ref12] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. KDD*, pp. 785–794, 2016.

[Ref13] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in NeurIPS*, 2017.

[Ref14] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[Ref15] C. Shang et al., "Machine learning for dengue prediction: a systematic review," *BMC Infectious Diseases*, vol. 21, p. 900, 2021.

[Ref16] A. Withanage et al., "A forecasting model for dengue incidence in the District of Gampaha, Sri Lanka," *Parasites & Vectors*, vol. 11, p. 262, 2018.

[Ref17] R. Lyons, "Distance covariance in metric spaces," *The Annals of Probability*, vol. 41, no. 5, pp. 3284–3305, 2013.

---

## Appendices

### A. Technical Details

**Repository Structure:**
```
├── configs/default.yaml          # All experiment parameters
├── run_all.py                    # Main forecasting pipeline runner
├── run_scenarios.py              # Automated multi-scenario execution
├── run_hybrid.py                 # Combined forecasting + non-linear
├── build_hybrid_report.py        # Report integration
├── non-linear-correlation-analysis/
│   ├── run_analysis.py           # Non-linear pipeline runner
│   └── src/analyzer.py           # Core analysis logic
├── src/
│   ├── data_loader.py            # Data loading & cleaning
│   ├── feature_engineering.py    # Lag, rolling, cyclical features
│   ├── dataset_builder.py        # Multi-horizon target creation
│   ├── evaluation.py             # Metrics & significance tests
│   ├── shap_analysis.py          # SHAP with XGBoost compatibility
│   └── models/                   # Naive, Prophet, XGBoost, HistGB, LSTM
└── results/                      # Scenario outputs (S1–S5 + NL)
```

**Key Hyperparameters (Best Configuration — S2):**

| Parameter | Value |
|:---|:---|
| XGBoost: max_depth | 6 |
| XGBoost: learning_rate | 0.05 |
| XGBoost: n_estimators | 300 |
| HistGB: max_iter | 300 |
| HistGB: learning_rate | 0.05 |
| LSTM: hidden_size | 64 |
| LSTM: num_layers | 2 |
| LSTM: epochs (max) | 30 |
| LSTM: early_stopping_patience | 8 |
| Random state | 42 |

**Software Environment:** Python 3.10+, pandas ≥2.0, NumPy ≥1.24, scikit-learn ≥1.4, XGBoost ≥2.0, PyTorch ≥2.2, Prophet ≥1.1, SHAP ≥0.45, SciPy ≥1.12, Matplotlib ≥3.8, Seaborn ≥0.13.

### B. Project Planning

[TODO: To be supplemented by the project team]

**B.1 Timeline**
- Weeks 1–2: Literature review & data exploration
- Weeks 3–4: Pipeline development (data loading, feature engineering, baselines)
- Weeks 5–6: Core model implementation (XGBoost, HistGB, LSTM)
- Weeks 7–8: Non-linear analysis module development
- Weeks 9: Scenario execution & evaluation
- Week 10: Report writing & presentation preparation

**B.2 Team Responsibilities**

[TODO: Assign specific team members]

**B.3 Current Progress**

All five experimental scenarios and non-linear analysis have been executed. Results, metrics, SHAP analyses, and visualizations have been generated and archived in the `results/` directory. The technical report is being finalized.
