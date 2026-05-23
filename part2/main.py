import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import build_pipeline
from model_comparison import model1_ols_full, model2_ols_selected, model3_ridge_lasso
from advanced_methods import model4_polynomial, model5a_kernel_svr, model5b_bayesian_ridge

if __name__ == '__main__':
    print("=" * 60)
    print("PIPELINE: Load & preprocess data")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, features = build_pipeline()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (OLS / Ridge / Lasso)")
    print("=" * 60)
    model1_ols_full(X_train, X_test, y_train, y_test)
    _, selected_features = model2_ols_selected(X_train, X_test, y_train, y_test)
    model3_ridge_lasso(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("ADVANCED METHODS (Polynomial / SVR / Bayesian)")
    print("=" * 60)
    model4_polynomial(X_train, X_test, y_train, y_test, degree=2)
    model5a_kernel_svr(X_train, X_test, y_train, y_test)
    model5b_bayesian_ridge(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
