# Data Fitting và Phương Pháp OLS

Đồ án 2 môn Toán Ứng Dụng và Thống Kê.

Repo gồm 2 phần chính:

- `part1`: trình bày lý thuyết, cài đặt OLS từ đầu và minh họa bằng dữ liệu giả lập.
- `part2`: ứng dụng data fitting trên bộ dữ liệu Stack Overflow Developer Survey 2023 để dự đoán lương developer.

## Cấu Trúc Thư Mục

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   `-- survey_results_public.csv
|-- part1/
|   |-- ols_implementation.py
|   |-- ridge_lasso.py
|   |-- residual_analysis.py
|   |-- cross_validation.py
|   |-- unit_test_1_4.py
|   |-- unit_test_5_9.py
|   `-- part1_notebook.ipynb
|-- part2/
|   |-- data_pipeline.py
|   |-- model_comparison.py
|   |-- advanced_methods.py
|   |-- generate_figures.py
|   |-- main.py
|   `-- part2_notebook.ipynb
`-- report/
    |-- report.pdf
    `-- report_overleaf/
        |-- main.tex
        |-- content/
        `-- images/
```

## Cài Đặt Môi Trường

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate script, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dữ Liệu

Bộ dữ liệu dùng trong Part 2:

- Tên file: `data/survey_results_public.csv`
- Nguồn: Stack Overflow Developer Survey 2023
- Link: https://survey.stackoverflow.co/2023/
- Link data: https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2023-developers-survey
- Kích thước gốc: 89,184 dòng x 84 cột
- Biến mục tiêu: `ConvertedCompYearly`

Lưu ý: file CSV gốc có dung lượng lớn nên không được đưa lên GitHub. Để chạy lại Part 2, cần tải bộ dữ liệu từ link trên, giải nén và đặt file `survey_results_public.csv` vào thư mục `data/`.

## Chạy Part 1

Chạy unit tests cho các nhóm hàm OLS, hat matrix, metrics, inference, VIF, Ridge, residual plots và k-fold CV:

```powershell
python part1\unit_test_1_4.py
python part1\unit_test_5_9.py
```

Mở notebook trình bày Part 1:

```powershell
jupyter notebook part1\part1_notebook.ipynb
```

## Chạy Part 2

Chạy riêng pipeline tiền xử lý:

```powershell
python part2\data_pipeline.py
```

Chạy toàn bộ pipeline Part 2, gồm load data, tiền xử lý, huấn luyện, đánh giá và advanced methods:

```powershell
python part2\main.py
```

Tạo lại các hình bằng script hiện có:

```powershell
python part2\generate_figures.py
```

Lưu ý: LaTeX trong `report/report_overleaf` đang include hình từ `report/report_overleaf/images/`. Nếu sinh lại hình bằng script, cần đảm bảo hình cuối cùng được đặt đúng thư mục mà file `.tex` đang tham chiếu.

Mở notebook trình bày Part 2:

```powershell
jupyter notebook part2\part2_notebook.ipynb
```

## Các Hàm Chính

Part 1:

- `ols_fit(X, y)`: ước lượng hệ số OLS và phương sai nhiễu.
- `hat_matrix(X)`: tính projection/hat matrix và kiểm tra tính idempotent.
- `model_metrics(y, y_hat, p)`: tính RSS, TSS, R2, adjusted R2 và F-statistic.
- `coef_inference(...)`: tính standard errors, t-statistics, p-values và confidence intervals.
- `vif(X)`: tính Variance Inflation Factor.
- `ridge_fit(X, y, lam)`: cài đặt Ridge Regression.
- `residual_plots(X, y, beta_hat)`: vẽ các biểu đồ chẩn đoán phần dư.
- `kfold_cv(X, y, k)`: cài đặt k-fold cross-validation.

Part 2:

- `DataPipeline`: xử lý missing values, encoding và standardization theo API `fit/transform`.
- `model_comparison.py`: so sánh OLS Full, OLS Selected, Ridge và Lasso.
- `advanced_methods.py`: Polynomial Features + Ridge và Bayesian Ridge.

## Kết Quả Part 2 Theo Báo Cáo

Metric được tính trên test set, với target ở thang `log1p(ConvertedCompYearly)`.

| Hạng | Mô hình | MAE | RMSE | R2 |
|---:|---|---:|---:|---:|
| 1 | Polynomial + Ridge | 0.5003 | 0.9090 | 0.4309 |
| 2 | OLS Full | 0.5378 | 0.9452 | 0.3847 |
| 3 | Ridge | 0.5378 | 0.9452 | 0.3847 |
| 4 | Bayesian Ridge | 0.5379 | 0.9452 | 0.3846 |
| 5 | OLS Selected | 0.5389 | 0.9456 | 0.3842 |
| 6 | Lasso | 0.5407 | 0.9467 | 0.3827 |

## Báo Cáo

File PDF nộp bài:

```text
report/report.pdf
```

Source LaTeX:

```text
report/report_overleaf/main.tex
```

## Ghi Chú Tái Lập Kết Quả

- Train/test split dùng `random_state=42`.
- Part 2 dùng dataset trong thư mục `data/`.
- Cần cài đầy đủ dependencies trong `requirements.txt` trước khi chạy test hoặc notebook.
