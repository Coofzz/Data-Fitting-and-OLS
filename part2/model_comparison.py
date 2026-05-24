"""
Model building, evaluation, and diagnostic plots for Part 2.

Models compared:
  M1 - OLS Full      : all features after preprocessing
  M2 - OLS Selected  : subset chosen by p-value (< 0.05) then VIF (<= 10)
  M3 - Ridge         : L2 regularization, alpha chosen by 5-fold CV
  M3 - Lasso         : L1 regularization, alpha chosen by 5-fold CV
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV, LassoCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import build_pipeline


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def compute_metrics(name: str, y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R² and print one summary row."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<38s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with VIF for each feature, sorted descending."""
    X_const = sm.add_constant(X)
    vif_df = pd.DataFrame({
        'feature': X.columns,
        'VIF': [
            variance_inflation_factor(X_const.values.astype(float), i + 1)
            for i in range(len(X.columns))
        ]
    })
    return vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def residual_plots(y_true, y_pred, model_name: str = 'Model') -> None:
    """
    Draw 4 standard regression diagnostic plots:
      1. Residuals vs Fitted
      2. Normal Q-Q Plot
      3. Scale-Location (sqrt|residuals| vs fitted)
      4. Residual Distribution Histogram
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Residual Diagnostic Plots — {model_name}', fontsize=14, fontweight='bold')

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.25, s=8, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Fitted values (log salary)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')

    # 2. Q-Q Plot (normality check)
    ax = axes[0, 1]
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title('Normal Q-Q Plot')

    # 3. Scale-Location (homoscedasticity)
    ax = axes[1, 0]
    ax.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.25, s=8, color='darkorange')
    ax.set_xlabel('Fitted values (log salary)')
    ax.set_ylabel('√|Residuals|')
    ax.set_title('Scale-Location (Homoscedasticity Check)')

    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(x_range,
            stats.norm.pdf(x_range, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max()-residuals.min()) / 60,
            color='red', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk on a sample (max 5000 for speed)
    sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
    stat, pval = stats.shapiro(sample)
    print(f"Residual stats — mean: {residuals.mean():.5f}  std: {residuals.std():.4f}")
    print(f"Shapiro-Wilk normality test (n={len(sample)}): W={stat:.4f}  p={pval:.4f}  "
          f"-> {'residuals appear normal' if pval > 0.05 else 'residuals deviate from normal'}")


def feature_importance_plot(coef, feature_names, model_name: str = 'Model', top_n: int = 20) -> None:
    """Plot horizontal bar chart of top-N absolute coefficients (standardized scale)."""
    coef_series = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    coef_series.sort_values().plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('|Coefficient| (standardized scale)')
    ax.set_title(f'Feature Importance (Top {top_n}) — {model_name}')
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 most influential features ({model_name}):")
    print(coef_series.head(10).to_string())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model1_ols_full(X_train, X_test, y_train, y_test):
    """M1: OLS regression with all features."""
    print("\n" + "="*60)
    print("M1: OLS Full (all features)")
    print("="*60)
    X_tr = sm.add_constant(X_train)
    X_te = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_tr).fit()
    print(model.summary())
    y_pred = model.predict(X_te)
    result = compute_metrics('M1 OLS Full', np.array(y_test), np.array(y_pred))
    return result, model, np.array(y_pred)


def model2_ols_selected(X_train, X_test, y_train, y_test):
    """M2: OLS with feature selection via p-value (< 0.05) then VIF (<= 10)."""
    print("\n" + "="*60)
    print("M2: OLS Selected (p-value + VIF filtering)")
    print("="*60)
    X_tr = sm.add_constant(X_train)
    m1   = sm.OLS(y_train, X_tr).fit()

    pvals     = m1.pvalues.drop('const')
    keep_pval = pvals[pvals < 0.05].index.tolist()
    print(f"Step 1 — p-value < 0.05: {len(keep_pval)}/{len(X_train.columns)} features retained")

    vif_df   = compute_vif(X_train[keep_pval])
    keep_vif = vif_df[vif_df['VIF'] <= 10]['feature'].tolist()
    print(f"Step 2 — VIF <= 10:      {len(keep_vif)} features retained")
    print(vif_df[vif_df['VIF'] > 5].to_string(index=False))

    X_tr2 = sm.add_constant(X_train[keep_vif])
    X_te2 = sm.add_constant(X_test[keep_vif])
    model = sm.OLS(y_train, X_tr2).fit()
    print(model.summary())

    y_pred = model.predict(X_te2)
    result = compute_metrics('M2 OLS Selected', np.array(y_test), np.array(y_pred))
    return result, model, np.array(y_pred), keep_vif


def model3_ridge_lasso(X_train, X_test, y_train, y_test):
    """M3: Ridge and Lasso with 5-fold CV for regularization parameter selection."""
    print("\n" + "="*60)
    print("M3: Ridge / Lasso (5-fold CV)")
    print("="*60)
    alphas = np.logspace(-3, 3, 100)

    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train, y_train)
    print(f"Ridge — best alpha: {ridge.alpha_:.4f}")
    y_pred_ridge = ridge.predict(X_test)
    r_ridge = compute_metrics('M3 Ridge', np.array(y_test), y_pred_ridge)

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    n_zero = (lasso.coef_ == 0).sum()
    print(f"Lasso — best alpha: {lasso.alpha_:.6f}  |  features zeroed: {n_zero}/{X_train.shape[1]}")
    y_pred_lasso = lasso.predict(X_test)
    r_lasso = compute_metrics('M3 Lasso', np.array(y_test), y_pred_lasso)

    return r_ridge, r_lasso, ridge, lasso, y_pred_ridge, y_pred_lasso


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def compare_models_table(results: list) -> pd.DataFrame:
    """Print a formatted comparison table sorted by R²."""
    df = pd.DataFrame(results)[['name', 'MAE', 'RMSE', 'R2']]
    df = df.sort_values('R2', ascending=False).reset_index(drop=True)
    df.index += 1
    print("\n" + "="*65)
    print("MODEL COMPARISON TABLE (sorted by R², evaluated on test set)")
    print("="*65)
    print(df.to_string())
    print("="*65)
    return df


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler, features = build_pipeline()

    r1, m1, p1     = model1_ols_full(X_train, X_test, y_train, y_test)
    r2, m2, p2, kv = model2_ols_selected(X_train, X_test, y_train, y_test)
    r3a, r3b, ridge, lasso, p3a, p3b = model3_ridge_lasso(X_train, X_test, y_train, y_test)

    compare_models_table([r1, r2, r3a, r3b])

    residual_plots(y_test, p3a, 'Ridge (best model)')
    feature_importance_plot(ridge.coef_, features, 'Ridge')
