# SCENARIOS — Multi-disease / config flags

## Variables
- Disease targets: `Influenza_rates`, `Dengue_fever_rates`, `Diarrhoea_rates`.
- Flags:
  - `compute_rate_per100k` (default false): if true, derive rate from `<cases_col>` and population; else use provided rate column directly.
  - `include_other_diseases_as_features` (default false): if true, add lag features of the other two diseases.

## Recommended run matrix (minimal but covers feedback)

| ID | Target              | compute_rate_per100k | include_other_diseases_as_features | Purpose |
|----|---------------------|-----------------------|------------------------------------|---------|
| S1 | Dengue_fever_rates  | false                 | false                               | Main baseline (current runs) |
| S2 | Dengue_fever_rates  | false                 | true                                | Check cross-disease predictive value |
| S3 | Dengue_fever_rates  | true (cases col set)  | false                               | Normalized-by-pop sensitivity |
| S4 | Influenza_rates     | false                 | false                               | Per-disease replication |
| S5 | Diarrhoea_rates     | false                 | false                               | Per-disease replication |

Notes:
- Run S2 only if muốn chứng minh “bệnh khác giúp dự báo” (có thể báo cáo thêm, không bắt buộc).
- Nếu thiếu `cases_col` cho Influenza/Diarrhoea, giữ `compute_rate_per100k=false`.
- Mỗi run chỉ cần đổi `target`, `cases_col`, và 2 cờ trong `configs/default.yaml`, rồi chạy `python run_all.py --config configs/default.yaml`.
