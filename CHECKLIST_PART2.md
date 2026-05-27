# Checklist Đối Chiếu Yêu Cầu — Part 2
> Đối chiếu với: `Toan UDTK_Project_2-Data Fitting va OLS.pdf`

---

## Mục 2.5 — Yêu Cầu Cài Đặt Python (6 mục)

| # | Yêu cầu đề bài | Đã làm | File / Section |
|---|---------------|--------|----------------|
| 1 | **`class DataPipeline`** xử lý missing values, encoding, chuẩn hóa theo thứ tự. Phải có thể `fit` trên train, `transform` trên test | ✅ Hoàn chỉnh | `data_pipeline.py` — class `DataPipeline` |
| 2 | **So sánh ≥ 3 mô hình**: Bảng tổng hợp MAE, RMSE, R² trên test set | ✅ 4 mô hình + 2 advanced | `model_comparison.py`, Notebook Section 8 |
| 3 | **Cross-validation** k-fold để chọn λ cho Ridge/Lasso (k=5 hoặc k=10) | ✅ 5-fold CV | `model_comparison.py:186` — `RidgeCV(cv=5)`, `LassoCV(cv=5)` |
| 4 | **Phân tích phần dư**: 4 biểu đồ chẩn đoán với mô hình tốt nhất | ✅ Đầy đủ | `model_comparison.py` — `residual_plots()`, Notebook Section 9 |
| 5 | **Feature importance**: Biểu đồ hệ số hồi quy (sau chuẩn hóa) | ✅ Hoàn chỉnh | `model_comparison.py` — `feature_importance_plot()`, Notebook Section 10 |
| 6 | **Nhận xét và kết luận**: Giải thích kết quả theo ngữ cảnh bộ dữ liệu | ✅ Chi tiết | Notebook Section 12 |

---

## Mục 2.6 — Tiêu Chí Đánh Giá (thang 5.5đ + 0.5đ bonus)

| Tiêu chí | Mô tả đề bài | Đã làm | Điểm | Đánh giá |
|----------|-------------|--------|------|----------|
| Chọn và mô tả dữ liệu | Đúng tiêu chí, mô tả rõ nguồn gốc | ✅ SO Survey 2023, n=89,184, p=10, có missing values, nguồn kaggle/stackoverflow | 0.5 | Đủ điều kiện |
| EDA | Đầy đủ thống kê mô tả, biểu đồ | ✅ Sections 1–6: describe, missing rate, histogram, boxplot, correlation heatmap, outlier | 0.5 | Đủ |
| Xử lý missing values | Đúng phương pháp, có giải thích | ✅ Phân tích MCAR/MAR/MNAR, listwise deletion có lý do | 1.0 | Đủ |
| Tiền xử lý tổng thể | Pipeline đầy đủ, fit/transform đúng | ✅ `DataPipeline` với `fit`/`transform`, không data leakage | 0.5 | Đủ |
| Xây dựng ≥ 3 mô hình | OLS, Ridge/Lasso, một mô hình khác | ✅ OLS Full, OLS Selected, Ridge, Lasso (4 mô hình) | 1.5 | Đủ |
| Đánh giá trên test set | MAE, RMSE, R², phân tích phần dư | ✅ Đủ 3 metrics + 4 residual plots + Shapiro-Wilk | 1.0 | Đủ |
| Nhận xét và kết luận | Phân tích có chiều sâu, liên hệ thực tế | ✅ Section 12 có limitations, model selection recommendation | 0.5 | Đủ |
| **Bonus** Kỹ thuật nâng cao | Kernel/Bayesian (tùy chọn) | ✅ Polynomial+Ridge (R²=0.431), Bayesian Ridge | +0.5 | Đã làm |

**Ước tính điểm Part 2: 5.5/5.5 + 0.5 bonus = 6.0/5.5**

---

## Mục 2.2 — Yêu Cầu EDA (Section 2.2.1)

| Yêu cầu | Đã làm | Section |
|---------|--------|---------|
| Thống kê mô tả: mean, median, std, min, max, quartiles | ✅ | Section 2 — `df.describe()` |
| Phân phối từng biến: histogram, boxplot | ✅ | Section 3 — histogram, Section 5 — boxplot |
| Ma trận tương quan: heatmap | ✅ | Section 4 — Pearson heatmap |
| Kiểm tra dữ liệu trùng lặp | ✅ `df.duplicated().sum() = 0` | Section 3 |
| Phân tích missing values: tỉ lệ thiếu theo cột | ✅ Bar chart + table | Section 3 |
| Phát hiện outliers: IQR, z-score | ✅ Cả hai phương pháp, so sánh | Section 5 |

---

## Mục 3.2 — Cấu Trúc Thư Mục Nộp Bài

```
Group_<ID>/
├── README.md                     ❌ CHƯA CÓ
├── requirements.txt              ❌ CHƯA CÓ
├── report/
│   ├── report.pdf                ❌ CHƯA CÓ (cần xuất từ .md sang PDF)
│   └── report.tex                ❌ Không bắt buộc nếu dùng Markdown
└── part2/
    ├── data/
    │   └── survey_results_public.csv  ✅ Có
    ├── data_pipeline.py               ✅ Có
    ├── model_comparison.py            ✅ Có
    ├── advanced_methods.py            ✅ Có (bonus)
    └── part2_notebook.ipynb           ✅ Có
```

---

## Mục 3.3 — Yêu Cầu Kỹ Thuật

| Yêu cầu | Trạng thái | Ghi chú |
|---------|-----------|---------|
| Python 3.10+ | ✅ | Python 3.14 |
| Code rõ ràng, chú thích nếu cần | ✅ | Docstrings có, comment tối giản |
| Biểu đồ có tiêu đề, nhãn trục, chú thích | ✅ | Đủ trong tất cả `plot()` calls |
| Mọi quyết định phải được giải thích | ✅ | MCAR/MAR/MNAR, VIF, p-value justified |
| Kết quả phải tái lập được (`random_state`) | ✅ | `random_state=42` ở tất cả chỗ |
| **Mỗi hàm có ít nhất 2 unit tests** | ✅ **30 tests (5/hàm × 6 hàm)** | Vượt yêu cầu |

---

## Mục 3.1 — Cấu Trúc Báo Cáo

| Mục | Yêu cầu | Trạng thái |
|-----|---------|-----------|
| Trang bìa | Họ tên, MSSV, nhóm, GVHD | ❌ Chưa có trong REPORT_PART2.md |
| Mục lục | Có mục lục | ❌ Chưa có |
| Phần 2 ứng dụng | Đầy đủ nội dung | ✅ REPORT_PART2.md |
| Kết luận | Tóm tắt, bài học, hướng mở rộng | ✅ Mục 6–7 trong báo cáo |
| Tài liệu tham khảo | Ít nhất 5 tài liệu | ✅ 8 tài liệu |
| Phụ lục | Bảng số liệu (nếu có) | — Không bắt buộc |

---

## Tóm Tắt: Những Gì CÒN THIẾU

| Hạng mục | Mức độ |
|----------|--------|
| `README.md` ở thư mục gốc nhóm | ⚠️ Cần có |
| `requirements.txt` (danh sách thư viện) | ⚠️ Cần có |
| `report.pdf` (xuất từ báo cáo .md) | ⚠️ Cần có khi nộp |
| Trang bìa trong báo cáo (họ tên, MSSV, nhóm) | ⚠️ Cần bổ sung |
| Mục lục trong báo cáo | ⚠️ Cần bổ sung |

---

## Kết Luận

**Phần code và notebook: ĐẦY ĐỦ** — tất cả 6 yêu cầu kỹ thuật của mục 2.5 đã được cài đặt, bonus section cũng có.

**Còn lại chủ yếu là hành chính nộp bài:**
1. Tạo `README.md` và `requirements.txt`
2. Bổ sung trang bìa + mục lục vào báo cáo
3. Xuất báo cáo ra PDF trước khi nộp
