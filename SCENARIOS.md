# SCENARIOS — Kịch Bản Chạy & Câu Hỏi Nghiên Cứu (Final Report Structure)

Để Báo cáo Cuối kỳ (Final Report) có tính mạch lạc, khoa học và phản hồi xuất sắc lại các nhận xét từ Giảng viên, các kịch bản chạy model (Runs) được tổ chức thành **3 bệnh × 2 lần thử** = **6 kịch bản dự báo** + **1 phân tích phi tuyến nền tảng**.

---

## Lưu ý quan trọng về dữ liệu

> **Biến `*_rates` trong dữ liệu gốc ĐÃ là tỷ lệ per 100.000 dân.** Không cần (và không nên) tự tính lại từ `cases / population`. Flag `compute_rate_per100k` luôn để `false`.

---

## Mạch Kịch Bản (Storyline Matrix)

| Kịch Bản | Target Bệnh | Cross-disease Features | Mục Tiêu Nghiên Cứu | Ý Nghĩa / Kết Luận Rút Ra |
| :--- | :--- | :---: | :--- | :--- |
| **Bản lề (NL)** | Tất cả 3 bệnh | — | Đo lường tính phi tuyến | Chứng minh quy luật khí hậu – dịch bệnh là phi tuyến; biện luận cho Gradient Boosting + MI |
| **S1** | `Dengue_fever_rates` | ❌ | MIMO 12→6 chỉ với khí hậu + xã hội | Backbone: ML có vượt trội Naive ở tầm nhìn xa? |
| **S2** | `Dengue_fever_rates` | ✅ | Thêm biến bệnh chéo (Influenza, Diarrhoea) | Tác động cross-disease lên Dengue qua SHAP |
| **S3** | `Influenza_rates` | ❌ | Tổng quát hóa pipeline sang Cúm | Pipeline có đa năng không? |
| **S4** | `Influenza_rates` | ✅ | Cúm + biến bệnh chéo | Cross-disease features có giá trị cho Cúm? |
| **S5** | `Diarrhoea_rates` | ❌ | Tổng quát hóa sang Tiêu chảy | Kiểm tra giới hạn của MIMO cho bệnh có autocorrelation lag-1 cực mạnh |
| **S6** | `Diarrhoea_rates` | ✅ | Tiêu chảy + biến bệnh chéo | Cross-disease features có cứu vãn được ML cho Diarrhoea? |

---

## Hướng Dẫn Kỹ Thuật (Cách Mở Khóa Từng Kịch Bản)

Mọi kịch bản đều được kiểm soát bởi file `configs/default.yaml`.

### Chạy toàn bộ (phi tuyến + 6 kịch bản) — 1 lệnh duy nhất:

```bash
python run_hybrid.py
```

### Chạy chọn lọc:

```bash
python run_hybrid.py --skip-nonlinear          # Chỉ chạy 6 kịch bản
python run_hybrid.py --skip-scenarios           # Chỉ chạy phi tuyến
python run_hybrid.py --scenarios 1 2            # Chỉ chạy S1 và S2
```

### Chạy riêng 1 kịch bản (cấu hình thủ công):

Chỉnh sửa `configs/default.yaml` rồi chạy:

```bash
python run_all.py --config configs/default.yaml
```

### Chạy riêng phân tích phi tuyến:

```bash
python run_nonlinear.py --config configs/default.yaml
```

Output sẽ nằm ở `outputs/nonlinear/`.

---

### Chi tiết cấu hình cho từng kịch bản:

#### S1 — Dengue (Backbone)
```yaml
experiment:
  target: Dengue_fever_rates
  cases_col: Dengue_fever_cases
  compute_rate_per100k: false      # Data đã là rate per 100k
  include_other_diseases_as_features: false
```

#### S2 — Dengue + Cross-disease
```yaml
experiment:
  target: Dengue_fever_rates
  cases_col: Dengue_fever_cases
  compute_rate_per100k: false
  include_other_diseases_as_features: true
```

#### S3 — Influenza (Generalization)
```yaml
experiment:
  target: Influenza_rates
  cases_col: Influenza_cases
  compute_rate_per100k: false
  include_other_diseases_as_features: false
```

#### S4 — Influenza + Cross-disease
```yaml
experiment:
  target: Influenza_rates
  cases_col: Influenza_cases
  compute_rate_per100k: false
  include_other_diseases_as_features: true
```

#### S5 — Diarrhoea (Generalization)
```yaml
experiment:
  target: Diarrhoea_rates
  cases_col: Diarrhoea_cases
  compute_rate_per100k: false
  include_other_diseases_as_features: false
```

#### S6 — Diarrhoea + Cross-disease
```yaml
experiment:
  target: Diarrhoea_rates
  cases_col: Diarrhoea_cases
  compute_rate_per100k: false
  include_other_diseases_as_features: true
```

---

## Thiết kế so sánh (Comparison Design)

Với 3 bệnh × 2 lần thử, ta có thể rút ra 2 loại insight:

1. **So sánh ngang (Cross-disease):** S1 vs S3 vs S5 — Pipeline tổng quát hóa tốt đến đâu?
2. **So sánh dọc (Ablation):** S1↔S2, S3↔S4, S5↔S6 — Biến bệnh chéo có giá trị dự báo bổ sung?

---

> **LỜI KHUYÊN CHO FINAL REPORT:** S1 + S2 và phân tích Phi tuyến là **xương sống** (70% báo cáo). S3-S6 là chương Thảo luận mở rộng (Discussion) thể hiện tính tổng quát và chiều sâu phân tích.
