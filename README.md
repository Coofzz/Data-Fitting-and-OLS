# Part 2 — Data Fitting và OLS

**Môn học:** Toán Ứng Dụng và Thống Kê  
**Đề tài:** Phân tích và Dự đoán Lương Developer — Stack Overflow Survey 2023

---

## Cấu trúc thư mục

```
part2/
├── README.md                    # File này
├── requirements.txt             # Danh sách thư viện
├── REPORT_PART2.md              # Báo cáo kỹ thuật đầy đủ
├── CHECKLIST_PART2.md           # Đối chiếu yêu cầu đề bài
│
├── report/
│   └── figures/                 # 9 biểu đồ PNG nhúng vào báo cáo
│       ├── fig1_missing_rate.png
│       ├── fig2_salary_transform.png
│       ├── fig3_outlier_detection.png
│       ├── fig4_correlation_heatmap.png
│       ├── fig5_pipeline_diagram.png
│       ├── fig6_model_comparison.png
│       ├── fig7_residual_plots.png
│       ├── fig8_feature_importance.png
│       └── fig9_ridge_alpha_cv.png
│
├── data/
│   └── survey_results_public.csv   # Stack Overflow Survey 2023
│
└── part2/                       # Source code
    ├── data_pipeline.py         # class DataPipeline + build_pipeline()
    ├── model_comparison.py      # OLS Full, OLS Selected, Ridge, Lasso
    ├── advanced_methods.py      # Polynomial+Ridge, Bayesian Ridge (bonus)
    ├── generate_figures.py      # Script tạo 9 biểu đồ cho báo cáo
    ├── main.py                  # Entry point — chạy toàn bộ pipeline
    └── part2_notebook.ipynb     # Jupyter notebook (13 sections)
```

---

## Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài thư viện
pip install -r requirements.txt
```

---

## Chạy code

```bash
# Activate venv trước, sau đó:
cd part2

# Chạy toàn bộ pipeline (load → preprocess → train → evaluate)
python main.py

# Tạo lại các biểu đồ cho báo cáo
python generate_figures.py
```

---

## Mô hình đã cài đặt

| # | Mô hình | R² (test) | MAE |
|---|---------|-----------|-----|
| 1 | OLS Full (38 features) | ~0.410 | ~0.472 |
| 2 | OLS Selected (p<0.05, VIF≤10) | ~0.409 | ~0.472 |
| 3 | Ridge (α=0.6136, 5-fold CV) | ~0.410 | ~0.472 |
| 4 | Lasso (α=0.001, 5-fold CV) | ~0.410 | ~0.472 |
| 5 | Polynomial (d=2) + Ridge | **~0.431** | ~0.460 |
| 6 | Bayesian Ridge | ~0.410 | ~0.473 |

---

## Dataset

- **Nguồn:** Stack Overflow Developer Survey 2023  
- **Link:** https://survey.stackoverflow.co/2023  
- **Kích thước gốc:** 89,184 quan sát × 84 cột  
- **Sau xử lý:** 48,019 hàng (sau drop NaN biến mục tiêu), 38,415 train / 9,604 test, 38 features

---

## Yêu cầu kỹ thuật đã đáp ứng

- [x] `class DataPipeline` với `fit()` / `transform()` API (không data leakage)
- [x] So sánh ≥ 3 mô hình với bảng MAE, RMSE, R²
- [x] 5-fold Cross-validation để chọn λ cho Ridge và Lasso
- [x] Phân tích phần dư: 4 biểu đồ + Shapiro-Wilk test
- [x] Biểu đồ feature importance (hệ số Ridge sau chuẩn hóa)
- [x] 30 unit tests (5 tests × 6 hàm) — vượt yêu cầu 2 tests/hàm
- [x] Kỹ thuật nâng cao: Polynomial Features + Bayesian Ridge (bonus +0.5đ)
