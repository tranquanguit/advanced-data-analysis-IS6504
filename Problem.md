# Nhật ký Vấn đề Kỹ thuật (Project Issues & Discrepancies)

Báo cáo này ghi nhận các điểm đặc biệt, hạn chế kỹ thuật hoặc các quyết định thiết kế "Proxy" trong hệ thống dự báo nhằm phục vụ việc giải trình trong báo cáo khoa học.

---

## 1. Giải thích mô hình thắng cuộc (SHAP for Winner)

### Trạng thái: ĐÃ XỬ LÝ (RESOLVED)
- **Vấn đề đã giải quyết:** Trước đây, SHAP luôn giải thích cho XGBoost ngay cả khi HistGB thắng cuộc. 
- **Giải pháp đã triển khai:** Đã cập nhật `run_all.py` để tự động xác định mô hình thắng cuộc (Winner) và truyền đối tượng mô hình đó vào phân tích SHAP. 
- **Tính tương thích:** Thư viện `shap_analysis.py` đã được nâng cấp để hỗ trợ đồng thời cả XGBoost, HistGB và LightGBM, bao gồm cả việc xử lý đầu ra đa ngưỡng (MIMO). 
- **Kết quả:** Các artifact XAI (SHAP plots, top features) hiện đã phản ánh đúng logic của mô hình đạt hiệu năng cao nhất, đảm bảo tính nhất quán tuyệt đối cho báo cáo khoa học.

---

## 2. Monkey-patching cho XGBoost >= 2.1.0

### Trạng thái: ĐÃ XỬ LÝ (RESOLVED)
- **Vấn đề:** SHAP bản 0.45 gặp lỗi khi giải mã định dạng `ubjson` từ XGBoost phiên bản mới.
- **Giải pháp:** Đã triển khai mã "Monkey-patch" trong `src/shap_analysis.py` để xử lý định dạng dữ liệu ngay trong bộ nhớ.

---

## 3. Mất mát dữ liệu tỉnh thành (Resolved)

### Trạng thái: ĐÃ XỬ LÝ (RESOLVED)
- **Vấn đề:** 7 tỉnh thành bị loại bỏ do lệnh `dropna()` quá hung hăng trên các cột không tham gia huấn luyện.
- **Giải pháp:** 
    1. **Lọc cột sớm (Early Filtering):** Loại bỏ các cột rác ngay sau khi load dữ liệu.
    2. **Xóa muộn có chọn lọc (Selective Dropna):** Chỉ xóa dòng nếu thiếu các biến quan trọng (Whitelist).
- **Kết quả:** Hệ thống đã khôi phục đầy đủ **55/55 tỉnh thành**. Tổng số mẫu huấn luyện đã tăng lên, đảm bảo tính đại diện cao nhất.

---

*Người ghi nhận: Antigravity AI Assistant*
*Ngày ghi nhận: 19/04/2026*
