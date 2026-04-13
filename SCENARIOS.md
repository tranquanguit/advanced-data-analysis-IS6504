# SCENARIOS — Kịch Bản Chạy & Câu Hỏi Nghiên Cứu (Final Report Structure)

Để Báo cáo Cuối kỳ (Final Report) có tính mạch lạc, khoa học và phản hồi xuất sắc lại các nhận xét từ Giảng viên, các kịch bản chạy model (Runs) không chỉ dừng lại ở liệt kê thông số tĩnh (S1-S5) như trước đây. Chúng được tái cấu trúc thành **4 Câu Hỏi Nghiên Cứu (Research Questions)**. 

Thay vì chỉ "chạy code để ra kết quả", mỗi kịch bản dưới đây mang một sứ mệnh bảo vệ một quan điểm học thuật.

---

## Mạch Kịch Bản (Storyline Matrix)

| Kịch Bản | Mục Tiêu Nghiên Cứu | Target Bệnh | Setting cần tinh chỉnh (`configs/default.yaml`) | Ý Nghĩa / Kết Luận Rút Ra |
| :--- | :--- | :--- | :--- | :--- |
| **Bản lề (Cơ sở)** | *Tại sao lại cần Machine Learning?* Đo lường tính phi tuyến. | `Dengue_fever_rates` | *Chỉ chạy file `run_analysis.py` trong thư mục non-linear* | Chứng minh quy luật khí hậu - dịch bệnh là phi tuyến (phân tán mạnh). Biện luận rằng dùng mô hình Tuyến tính/Truyền thống là thiếu sót, từ đó dọn đường cho thuật toán MI và Gradient Boosting. |
| **Kịch Bản 1 (S1)** | *Tầm nhìn xa:* Đột phá với MIMO (Multi-Input Multi-Output). | `Dengue_fever_rates` | `target: Dengue_fever_rates`<br>`input_sequence_length: 12`<br>`predict_horizon: 6`<br>`include_other_diseases...: false` | **Phản biện lại nhận xét "Bài toán Forecasting vô nghĩa"**. Khi kéo dài cửa sổ quá khứ 12 tháng dự báo lên tương lai 6 tháng, Naive bị đuối sức nghiêm trọng, trong khi XGBoost/HistGB thể hiện khả năng dự đoán cục diện dài hạn cực xuất sắc. |
| **Kịch Bản 2 (S2)** | *Sự lây nhiễm chéo:* Bệnh này có kéo theo bệnh kia? | `Dengue_fever_rates` | `target: Dengue_fever_rates`<br>`include_other_diseases...: true` | Tìm hiểu xem biến số của Cúm (Influenza) hay Tiêu chảy (Diarrhoea) có ý nghĩa dự báo (predictive value) cho Dengue Fever hay không bằng cách xem biểu đồ **SHAP values**. Giải thích được hệ sinh thái y tế. |
| **Kịch Bản 3 (S3)** | *Tính tổng quát hóa:* Siêu mô hình hay chỉ là may mắn Overfitting? | `Influenza_rates` | `target: Influenza_rates`<br>`include_other_diseases...: false` | Dùng nguyên kiến trúc MIMO ưu việt của Sốt xuất huyết áp dụng sang Cúm mùa. Nếu metrics vẫn tốt, khẳng định đây là một Pipeline đa năng có thể triển khai hệ thống cho Bộ Y Tế chặn ứng bất kỳ dịch bệnh nào. |
| **Kịch Bản 4 (S4)** | *Tính chuẩn hóa:* Độ nhạy cảm với Dân số (Population Normalization). | `Dengue_fever_rates` | `compute_rate_per100k: true`<br>`cases_col: Dengue_fever_cases` | Đánh giá xem mô hình có hoạt động khác đi khi mô hình tự linh suy quy đổi ra (cases / population) thay vì số rate mặc định có sẵn. |

---

## Hướng Dẫn Kỹ Thuật (Cách Mở Khóa Từng Kịch Bản)

Mọi kịch bản (trừ module phi tuyến nằm riêng) đều được kiểm soát bởi file `configs/default.yaml`.
Sau khi đổi cờ (flags), chỉ cần gõ duy nhất 1 lệnh tại thư mục gốc:

```bash
python run_all.py --config configs/default.yaml
```

### Chi tiết cách thay đổi cờ cho Báo Cáo:

1. **Chạy Kịch bản Bản Lề (Non-Linear Analysis):**
   ```bash
   cd non-linear-correlation-analysis
   python run_analysis.py
   ```
   *Lấy ảnh Heatmap sinh ra ở `outputs/plots/` để viết phần Cơ sở Dữ liệu của Báo cáo.*

2. **Chạy Kịch Bản Tầm nhìn xa MIMO (S1 - Backbone):**
   Trong file YAML, chỉnh sửa phần `experiment` & `run`:
   ```yaml
   experiment:
     target: Dengue_fever_rates
     compute_rate_per100k: false
     include_other_diseases_as_features: false
     input_sequence_length: 12   # Lịch sử 1 năm
     predict_horizon: 6          # Tầm nhìn 6 tháng
   ```
   *Thu thập `model_comparison.csv` và ảnh dự báo của HistGB để làm điểm nhấn Báo cáo.*

3. **Chạy Kịch Bản Lây nhiễm chéo (S2 - SHAP Insights):**
   Đổi cờ `include_other...`:
   ```yaml
   experiment:
     target: Dengue_fever_rates
     include_other_diseases_as_features: true
   ```
   *Thu thập file `outputs/shap/insight.txt` và `top_features.csv` để xem mức độ quan trọng.*

4. **Chạy Kịch Bản Tính Tổng Quát (S3 - Scale up):**
   Đổi mục tiêu học máy sang `Influenza_rates` hoặc `Diarrhoea_rates`:
   ```yaml
   experiment:
     target: Influenza_rates
     include_other_diseases_as_features: false
   ```
   *So sánh bảng metric để kết luận.*

---
> **LỜI KHUYÊN CHO FINAL REPORT:** Không nhất thiết phải ốp toàn bộ output của 4 kịch bản này vào slide một cách ngang hàng. S1 và Lập luận Phi tuyến tính là Xương sống (Chiếm 60% hàm lượng báo cáo). S2 và S3 là các Chapter Thảo luận mở rộng (Discussion) nhằm thể hiện độ sâu sắc, cái tâm của người làm Research Data.
