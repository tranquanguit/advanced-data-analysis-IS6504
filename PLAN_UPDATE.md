# PLAN_UPDATE — Nâng cấp Pipeline hướng Paper-ready

## Tổng quan

Kế hoạch 5 bước nâng cấp pipeline từ mức đồ án Master lên mức có thể submit paper workshop/regional journal. Mỗi bước được thiết kế để **tăng chất lượng nghiên cứu một cách khách quan**, không gò ép kết quả.

---

## BƯỚC 1: NL-guided Selective Cross-disease Features (S7/S8/S9)

### Mô tả
Hiện tại S2/S4/S6 thêm **toàn bộ** lag 1–12 của 2 bệnh khác (+26 features). Phần lớn là noise, đặc biệt cho Dengue (không có cặp cross-disease nào trong top 30 NL). 

**Thay đổi:** Dùng kết quả phân tích phi tuyến (NL) để **chọn lọc** chỉ các (predictor, lag) có dCor > threshold.

### Bằng chứng từ NL analysis

| Target | Predictor | Lags có dCor > 0.4 | Lags có dCor > 0.45 |
|:---|:---|:---|:---|
| **Dengue** | Influenza, Diarrhoea | **Không có** (=0 lags) | 0 |
| **Influenza** | Diarrhoea | lag 0,1,2,3,4,5,6 (all >0.42) | lag 0,1,2,3 |
| **Diarrhoea** | Influenza | lag 0,1,3,4 (>0.41) | lag 0,1,3 |

### Kịch bản mới

| Kịch bản | Target | Cross-disease features | Số features thêm |
|:---|:---|:---|---:|
| **S7** | Dengue | Không thêm (NL xác nhận không có tín hiệu) | 0 |
| **S8** | Influenza | Chỉ `Diarrhoea_rates` lag 0–3 | +4 (thay vì +26) |
| **S9** | Diarrhoea | Chỉ `Influenza_rates` lag 0–1 | +2 (thay vì +26) |

### Tác động lên project

- **Code:** Thêm tham số `cross_disease_map` trong config, sửa `feature_engineering.py` để nhận dict `{predictor: [lags]}` thay vì boolean `include_other_diseases`.
- **File ảnh hưởng:** `configs/default.yaml`, `src/feature_engineering.py`, `run_all.py`, `run_hybrid.py`
- **Kỳ vọng:** 
  - Dengue S7 = S1 (không thêm gì, vì NL xác nhận không nên thêm) → trung thực
  - Influenza S8: improvement > S4 (ít noise hơn, ít features hơn)
  - Diarrhoea S9: improvement > S6 (ít noise hơn)
- **Thời gian:** 1–2 ngày

### Tính khách quan
**Cao.** Quyết định thêm/không thêm dựa 100% trên phân tích NL độc lập (chạy trước forecasting), không phải nhìn kết quả rồi chỉnh ngược. Pipeline NL là bước đánh giá khách quan → kết quả NL guide feature selection cho forecasting. Đây là **contribution chính** nếu viết paper: "NL analysis as feature selection guide".

---

## BƯỚC 2: Weighted MIMO Loss

### Mô tả
Hiện tại MIMO tối ưu MSE/MAE trung bình 6 horizon → hy sinh h=1 để cải thiện h=3–6. Đặc biệt gây bất lợi cho Diarrhoea (Naive thắng ML ở h=1).

**Thay đổi:** Cho phép config trọng số theo horizon:

```yaml
experiment:
  horizon_weights: [2.0, 1.5, 1.2, 1.0, 0.8, 0.7]  # ưu tiên ngắn hạn
  # hoặc: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # đều (hiện tại)
```

### Tác động lên project

- **Code:** Sửa loss function trong `src/trainer.py` (LSTM) và custom objective cho XGBoost/HistGB (hoặc dùng `sample_weight` theo horizon).
- **File ảnh hưởng:** `src/trainer.py`, `src/models/tree_models.py`, `configs/default.yaml`
- **Kỳ vọng:**
  - Diarrhoea: ML có thể vượt Naive ở h=1 → loại bỏ "negative result"
  - Dengue: Cross-disease penalty ở h=1 giảm (vì h=1 được ưu tiên)
  - Trade-off: MAE@5,6 có thể tăng nhẹ
- **Thời gian:** 0.5 ngày

### Tính khách quan
**Trung bình.** Weighted loss là kỹ thuật standard (nhiều paper cũng dùng), nhưng cần báo cáo **cả hai cấu hình** (uniform vs weighted) — không chỉ chọn cái nào cho kết quả đẹp hơn. Nếu chỉ báo weighted mà giấu uniform → mất khách quan.

**Lưu ý:** Nên chạy cả 2 weight configs và so sánh, trình bày như ablation study.

---

## BƯỚC 3: Rolling Cross-validation (5 folds)

### Mô tả
Hiện tại chỉ có 1 train/val/test split cố định. Reviewer sẽ hỏi: "Kết quả có robust không hay chỉ may mắn với split này?"

**Thay đổi:** Expanding-window cross-validation:

```
Fold 1: Train ≤2012, Val 2013, Test 2014
Fold 2: Train ≤2013, Val 2014, Test 2015
Fold 3: Train ≤2014, Val 2015, Test 2016
Fold 4: Train ≤2015, Val 2016, Test 2017
Fold 5: Train ≤2016, Val 2017, Test 2018
```

### Tác động lên project

- **Code:** Thêm loop ngoài cùng trong `run_all.py` hoặc tạo `run_cv.py` mới.
- **File ảnh hưởng:** `run_all.py` (hoặc file mới `run_cv.py`), `EXPERIMENTS_AND_RESULTS.md`
- **Kỳ vọng:**
  - Kết quả: `MAE@h = mean ± std` qua 5 folds
  - Nếu cross-disease cải thiện ≥4/5 folds → bằng chứng rất mạnh
  - Nếu chỉ 2–3/5 folds → cần báo cáo trung thực là "mixed results"
- **Thời gian:** 2–3 ngày (chạy pipeline × 5 folds × 6 scenarios = 30 runs ≈ 6 giờ)

### Tính khách quan
**Rất cao.** Cross-validation là gold standard trong ML research. Bất kể kết quả tốt hay xấu, nó **tăng credibility** cho báo cáo. Nếu kết quả xấu ở một số fold, đó cũng là phát hiện có giá trị.

---

## BƯỚC 4: Thêm LightGBM (model bổ sung)

### Mô tả
Hiện chỉ có XGBoost + HistGradientBoosting (cùng họ gradient boosting). Reviewer có thể hỏi: "Tại sao không thử LightGBM — baseline chuẩn nhất hiện tại cho tabular data?"

**Thay đổi:** Thêm `LightGBM` vào pipeline song song XGBoost/HistGB (cùng grid search pattern).

### Tác động lên project

- **Code:** Thêm `src/models/lgbm_model.py`, sửa `run_all.py` thêm 1 model.
- **File ảnh hưởng:** `src/models/lgbm_model.py` (mới), `run_all.py`
- **Kỳ vọng:** LightGBM thường ngang/nhỉnh hơn HistGB trên tabular data. Nếu kết quả tương đương → xác nhận HistGB là lựa chọn tốt. Nếu LightGBM tốt hơn → có thêm 1 model mạnh.
- **Thời gian:** 0.5–1 ngày

### Tính khách quan
**Cao.** Thêm model không bias kết quả theo hướng nào — chỉ mở rộng phạm vi so sánh.

---

## BƯỚC 5: Wilcoxon Test giữa Base vs Cross-disease trực tiếp

### Mô tả
Hiện tại Wilcoxon chỉ so sánh mỗi model vs Seasonal Naive. Câu hỏi nghiên cứu "Cross-disease có cải thiện không?" cần test **trực tiếp** S1 vs S2, S3 vs S4, S5 vs S6.

**Thay đổi:** Thêm kiểm định paired Wilcoxon giữa absolute errors của HistGB-S1 vs HistGB-S2 tại mỗi horizon.

### Tác động lên project

- **Code:** Thêm function trong `src/evaluation.py`, gọi trong `run_hybrid.py` sau khi chạy xong các cặp scenario.
- **File ảnh hưởng:** `src/evaluation.py`, `run_hybrid.py` (post-processing)
- **Kỳ vọng:**
  - Nếu p < 0.05 cho S3→S4 hay S5→S6 tại h=3+ → bằng chứng thống kê mạnh
  - Nếu p > 0.05 → cải thiện chỉ là xu hướng, chưa đạt ý nghĩa thống kê → báo cáo trung thực
- **Thời gian:** 0.5 ngày

### Tính khách quan
**Rất cao.** Đây là cách đúng đắn duy nhất để trả lời câu hỏi "có cải thiện thống kê hay không". Kết quả p > 0.05 cũng là kết quả có giá trị.

---

## Tóm tắt tác động và ưu tiên

| Bước | Effort | Impact trên kết quả | Tính khách quan | Ưu tiên |
|:---|:---|:---|:---|:---|
| **1. NL-guided features** | 1–2 ngày | 🔴🔴🔴 Cao nhất | ✅ Rất cao | ⭐⭐⭐ |
| **2. Weighted MIMO** | 0.5 ngày | 🔴🔴 Cao | 🟡 Trung bình | ⭐⭐ |
| **3. Rolling CV** | 2–3 ngày | 🔴🔴🔴 Bắt buộc cho paper | ✅ Rất cao | ⭐⭐⭐ |
| **4. LightGBM** | 0.5–1 ngày | 🟡 Trung bình | ✅ Cao | ⭐ |
| **5. Wilcoxon cross** | 0.5 ngày | 🔴🔴 Cao | ✅ Rất cao | ⭐⭐ |
| **Tổng** | **5–7 ngày** | | | |

---

## Decision Matrix: Chạy hay không?

| Nếu... | Thì kết luận |
|:---|:---|
| Bước 1 cho improvement -5% đến -10% | Paper story rất mạnh: "NL guides feature selection" |
| Bước 1 cho improvement < -2% | NL guides feature selection có hiệu quả nhẹ; cần rolling CV để strengthen |
| Bước 3 cho 4+/5 folds nhất quán | Robust, publishable |
| Bước 3 cho 2–3/5 folds | Mixed results, trung thực báo cáo, vẫn có giá trị |
| Bước 5 cho p > 0.05 | "Xu hướng cải thiện nhưng chưa đạt ý nghĩa" — vẫn publishable nếu kết hợp SHAP + NL |
