"""
Advanced regression methods for Part 2 (bonus section).

M4 — Polynomial features (degree=2) + Ridge CV
M5b — Bayesian Ridge (uncertainty quantification)
"""

import os, sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import build_pipeline


def _compute_metrics(name: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<38s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def model4_polynomial(X_train, X_test, y_train, y_test, degree=2):
    """M4: Polynomial features (degree) + Ridge with 5-fold CV."""
    print(f"\n── M4: Polynomial (degree={degree}) + Ridge ──")

    poly       = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr_poly  = poly.fit_transform(X_train)
    X_te_poly  = poly.transform(X_test)
    print(f"Features: {X_train.shape[1]} -> {X_tr_poly.shape[1]} after polynomial expansion")

    model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    model.fit(X_tr_poly, y_train)
    print(f"Best alpha: {model.alpha_:.4f}")

    y_pred  = model.predict(X_te_poly)
    result  = _compute_metrics('M4 Polynomial+Ridge', np.asarray(y_test), y_pred)
    return result, model


def model5b_bayesian_ridge(X_train, X_test, y_train, y_test):
    """M5b: Bayesian Ridge — point predictions + predictive uncertainty (std)."""
    print("\n── M5b: Bayesian Ridge ──")

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_pred, y_std = model.predict(X_test, return_std=True)
    print(f"Mean predictive uncertainty (std): {y_std.mean():.4f}")

    result = _compute_metrics('M5b Bayesian Ridge', np.asarray(y_test), y_pred)
    return result, model, y_pred, y_std


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X_train, X_test, y_train, y_test, pipeline, features = build_pipeline()

    r4, poly_model                         = model4_polynomial(X_train, X_test, y_train, y_test, degree=2)
    r5, bayes_model, y_pred_bayes, y_std_b = model5b_bayesian_ridge(X_train, X_test, y_train, y_test)
