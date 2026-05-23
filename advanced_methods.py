import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.svm import SVR

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import build_pipeline


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"{name:35s}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return rmse, r2


def model4_polynomial(X_train, X_test, y_train, y_test, degree=2):
    print(f"\n── M4: Polynomial (degree={degree}) ──")

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)

    print(f"Features gốc: {X_train.shape[1]}  →  Sau polynomial: {X_tr_poly.shape[1]}")

    model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    model.fit(X_tr_poly, y_train)
    print(f"Best alpha: {model.alpha_:.4f}")

    y_pred = model.predict(X_te_poly)
    return evaluate("M4 Polynomial+Ridge", y_test, y_pred)


def model5a_kernel_svr(X_train, X_test, y_train, y_test):
    print("\n── M5a: Kernel Regression (SVR) ──")
    print("(SVR trên dữ liệu lớn có thể mất vài phút...)")

    model = SVR(kernel='rbf', C=10, epsilon=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return evaluate("M5a SVR (RBF kernel)", y_test, y_pred)


def model5b_bayesian_ridge(X_train, X_test, y_train, y_test):
    print("\n── M5b: Bayesian Ridge ──")

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_pred, y_std = model.predict(X_test, return_std=True)
    print(f"Độ không chắc chắn trung bình (std): {y_std.mean():.4f}")

    return evaluate("M5b Bayesian Ridge", y_test, y_pred)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    X_train, X_test, y_train, y_test, scaler, features = build_pipeline()

    model4_polynomial(X_train, X_test, y_train, y_test, degree=2)
    model5a_kernel_svr(X_train, X_test, y_train, y_test)
    model5b_bayesian_ridge(X_train, X_test, y_train, y_test)
