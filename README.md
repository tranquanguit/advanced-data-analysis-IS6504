# Infectious Disease Forecasting Research Pipeline (Automated Hybrid System)

A state-of-the-art research pipeline for multi-province, multi-horizon (MIMO) forecasting of **Dengue**, **Influenza**, and **Diarrhoea** in Vietnam. This system implements an advanced **Hybrid Architecture** that uses non-linear analysis to guide adaptive feature engineering, ensuring robust performance and academic rigor.

## 🚀 Key Scientific Features

- **Automated NL-Guided Selection:** Uses Distance Correlation (dCor) to identify significant lags across diseases, filtering out noise and secondary predictors dynamically.
- **Multi-Horizon MIMO Architecture:** Predicts 6 months simultaneously ($t+1 \dots t+6$) using specialized multi-output regression models.
- **Weighted Loss Optimization:** Prioritizes forecast accuracy for immediate horizons ($h=1, 2$) using a custom weighted MSE loss function.
- **Paired Statistical Testing:** Built-in Wilcoxon Signed-Rank Test to validate improvements between experimental scenarios.
- **Model Explainability (XAI):** Integrated SHAP analysis for global and provincial feature importance.

---

## 📂 Project Structure

```text
.
├── configs/              # Central configuration (YAML)
├── data/                 # Raw and processed datasets
├── outputs/              # Analysis artifacts (Plots, Tables, SHAP)
├── results/              # Final benchmarking metrics for S1-S9
├── src/                  # Core logic and source code
│   ├── models/           # HGB, LightGBM, XGBoost, LSTM, Prophet
│   ├── data_loader.py    # Robust data ingestion and cleaning
│   ├── feature_engineering.py # Selective lags & rolling stats
│   ├── nonlinear_analyzer.py  # dCor & MI metric computation
│   └── trainer.py        # Weighted MIMO optimization logic
├── run_hybrid.py         # Unified orchestrator (The primary entry point)
├── run_nonlinear.py      # Independent analysis runner (Step 1)
└── run_all.py            # Individual scenario runner (Step 2)
```

---

## 📊 Experimental Scenarios (S1 - S9)

The project systematically compares three feature selection strategies across three target diseases:

| Group | Scenarios | Target Strategy | Purpose |
| :--- | :--- | :--- | :--- |
| **Baseline** | S1, S3, S5 | Climate + Social Only | Define the minimal forecasting capability. |
| **Brute-force** | S2, S4, S6 | All lags (1-12) of other diseases | Test full cross-disease integration (often noisy). |
| **NL-Guided** | S7, S8, S9 | Selective lags (dCor > 0.4) | **Innovation:** Optimized selection via MI/dCor. |

---

## 🛠️ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Execution (The Unified Way)
To run the full research workflow (Analysis + All 9 Scenarios + Statistical Tests):
```bash
python run_hybrid.py
```

### 3. CLI Options
- **Skip Analysis:** `python run_hybrid.py --skip-nonlinear`
- **Run specific scenarios:** `python run_hybrid.py --scenarios 7 8 9`
- **Analysis only:** `python run_hybrid.py --skip-scenarios`

---

## 📈 Analysis & Outputs

- **Core Metrics:** `results/scenarioX/outputs/metrics/model_comparison.csv`
- **Statistical Significance:** `results/scenarioX/outputs/metrics/wilcoxon_vs_scenarioY.csv`
- **Distance Correlation Ranking:** `outputs/nonlinear/tables/global_lag_metrics_ranked.csv`
- **SHAP Feature Importance:** `results/scenarioX/outputs/shap/top_features.csv`

---

## ⚙️ Configuration

All system parameters are centralized in `configs/default.yaml`. You can adjust prediction horizons, model hyperparameters, and data paths without modifying the source code.

> [!IMPORTANT]
> The **NL-Guided scenarios (S7-S9)** are dynamically updated at runtime based on the most recent non-linear analysis results stored in `outputs/nonlinear/tables/suggested_lags.json`.

---

## 📚 Methodology References
1. **Distance Correlation:** Scikit-learn based non-linear dependency estimation.
2. **MIMO Strategy:** Multi-Input Multi-Output vector forecasting.
3. **Statistical Validity:** Paired Wilcoxon Signed-Rank Test for error distributions.
