# Dengue Forecasting Research Pipeline (Multi-Province, Multi-Horizon)

Pipeline này triển khai đầy đủ các phase đã thống nhất:

- **Phase 1**: Data prep + baseline (Naive, Seasonal Naive, Prophet)
- **Phase 2**: EDA + lag/correlation
- **Phase 3**: Modeling (XGBoost, HistGB, LSTM)
- **Phase 4**: Evaluation (MAE/RMSE/SMAPE + outbreak)
- **Phase 5**: SHAP + insight extraction

## 1) Cấu trúc thư mục

```text
.
├── configs/
│   └── default.yaml      # chỉnh tham số mà không sửa code
├── data/
│   ├── raw/              # đặt các file .xlsx theo tỉnh
│   └── processed/
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   ├── plots/
│   └── shap/
├── src/
│   ├── runtime_config.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── dataset_builder.py
│   ├── evaluation.py
│   ├── eda.py
│   ├── insight_extractor.py
│   ├── shap_analysis.py
│   ├── trainer.py
│   ├── visualization.py
│   └── models/
│       ├── naive.py
│       ├── prophet_model.py
│       ├── tree_models.py
│       └── lstm_model.py
└── run_all.py
```

## 2) Cài đặt

```bash
pip install -r requirements.txt
```

## 3) Chuẩn bị dữ liệu

- Đặt file tỉnh vào `data/raw/` (vd: `squeezed_Hà Nội.xlsx`).
- Mỗi file cần có tối thiểu các cột:
  - `year`, `month`, `Dengue_fever_rates`
  - climate vars và socio vars theo `configs/default.yaml`

> Lưu ý: pipeline sẽ tự tạo các thư mục `data/raw`, `data/processed`, `outputs/*` nếu chưa có.

## 4) Chạy pipeline

```bash
python run_all.py --config configs/default.yaml
```

## 5) Chỉnh tham số không cần sửa code

Chỉnh trực tiếp trong `configs/default.yaml`:
- danh sách biến
- lag / rolling / horizons
- train/test split
- params XGB/HistGB/LSTM
- bật/tắt Prophet, SHAP

## 6) Output chính

- `outputs/metrics/model_comparison.csv`: MAE@1/2/3 theo model
- `outputs/metrics/province_metrics.csv`: MAE theo tỉnh
- `outputs/plots/*.png`: biểu đồ EDA và prediction
- `outputs/shap/top_features.csv`: xếp hạng feature
- `outputs/shap/insights.txt`: insight tự động

## 7) Checklist thực thi (assign team)

| Phase | Task | Model/Method | Scope | Key Params | Owner | Deadline (Day) | Expected Numeric Outcome | Expected Insight |
|---|---|---|---|---|---|---:|---|---|
| 1 | Chuẩn hóa dữ liệu, index (`province`,`date`) | Data pipeline | 53 tỉnh | monthly, sort asc, ffill+bfill | Data Lead | 1 | 100% file load pass | Không leak thời gian |
| 1 | Feature lag + rolling | FE | 53 tỉnh | lags=1,2,3; roll=3 mean/std | Data Lead | 2 | feature matrix đầy đủ | Lag hiệu lực 1-3 tháng |
| 1 | Baseline | Naive / Seasonal Naive / Prophet | Per-province | horizon=1,2,3 | Baseline Eng | 3 | Seasonal Naive > Naive | Seasonality rõ |
| 2 | EDA mùa vụ, CCF, phân bố | Plots + stats | National + region | month profile, corr by lag | Data Analyst | 5 | xác định lag tốt nhất | Rain/Humidity lag mạnh |
| 3 | Core model | XGB / HistGB / LSTM | Global pooled | XGB(300,6,0.05), LSTM(hidden=64) | ML + DL | 9 | tốt hơn baseline 5–10% | Global học được pattern chung |
| 3 | Input impact | LSTM | Global | disease-only / +climate / +socio | DL | 9 | +climate cải thiện 5–10% | Climate có predictive power |
| 3 | Lag impact | XGB | Global | lag_set=1,3,6 | ML | 9 | lag 2–3 tốt nhất | hiệu ứng trễ bệnh |
| 4 | Eval + significance | MAE/RMSE/SMAPE, Wilcoxon | All | horizons=1,2,3 | QA/Statistician | 10 | p-value < 0.05 | khác biệt có ý nghĩa |
| 5 | Explainability | SHAP global + per province | Global + tỉnh | mean(abs(shap)) | Explainability Eng | 11 | top climate lag features | khác biệt vùng |
| 5 | Report package | CSV + plots + insights | All | autosave output | Tech Lead | 12 | report-ready artifacts | claim contribution rõ ràng |

> Gợi ý nhân sự 5 người: Data Lead, Baseline Eng, ML Eng, DL Eng, QA/Statistician.
