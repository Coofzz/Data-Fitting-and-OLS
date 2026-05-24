import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV, LassoCV

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import build_pipeline


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"{name:30s}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return rmse, r2


def model1_ols_full(X_train, X_test, y_train, y_test):
    print("\n── M1: OLS Cơ Bản ──")

    X_tr = sm.add_constant(X_train)
    X_te = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_tr).fit()
    print(model.summary())

    y_pred = model.predict(X_te)
    return evaluate("M1 OLS full", y_test, y_pred)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_const.values, i + 1)
        for i in range(len(X.columns))
    ]
    return vif_data.sort_values('VIF', ascending=False)


def model2_ols_selected(X_train, X_test, y_train, y_test):
    print("\n── M2: OLS Chọn Biến ──")

    X_tr = sm.add_constant(X_train)
    m1   = sm.OLS(y_train, X_tr).fit()

    pvals = m1.pvalues.drop('const')
    keep_pval = pvals[pvals < 0.05].index.tolist()
    print(f"Giữ theo p-value (<0.05): {len(keep_pval)} features")

    vif_df = compute_vif(X_train[keep_pval])
    print(vif_df.to_string(index=False))

    keep_vif = vif_df[vif_df['VIF'] <= 10]['feature'].tolist()
    print(f"Giữ theo VIF (<=10): {len(keep_vif)} features")

    X_tr2 = sm.add_constant(X_train[keep_vif])
    X_te2 = sm.add_constant(X_test[keep_vif])
    model = sm.OLS(y_train, X_tr2).fit()
    print(model.summary())

    y_pred = model.predict(X_te2)
    return evaluate("M2 OLS selected", y_test, y_pred), keep_vif


def model3_ridge_lasso(X_train, X_test, y_train, y_test):
    print("\n── M3: Ridge / Lasso ──")

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge.fit(X_train, y_train)
    print(f"Ridge best alpha: {ridge.alpha_:.4f}")

    y_pred_ridge = ridge.predict(X_test)
    evaluate("M3 Ridge", y_test, y_pred_ridge)

    lasso = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000)
    lasso.fit(X_train, y_train)
    print(f"Lasso best alpha: {lasso.alpha_:.4f}")

    zero_coef = np.sum(lasso.coef_ == 0)
    print(f"Lasso loại {zero_coef}/{X_train.shape[1]} features (coef=0)")

    y_pred_lasso = lasso.predict(X_test)
    evaluate("M3 Lasso", y_test, y_pred_lasso)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    X_train, X_test, y_train, y_test, scaler, features = build_pipeline()

    model1_ols_full(X_train, X_test, y_train, y_test)
    _, selected_features = model2_ols_selected(X_train, X_test, y_train, y_test)
    model3_ridge_lasso(X_train, X_test, y_train, y_test)
