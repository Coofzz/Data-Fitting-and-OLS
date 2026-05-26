import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import build_pipeline
from model_comparison import (
    model1_ols_full, model2_ols_selected, model3_ridge_lasso,
    compare_models_table
)
from advanced_methods import model4_polynomial, model5b_bayesian_ridge

if __name__ == '__main__':
    print("=" * 60)
    print("PIPELINE: Load & preprocess data")
    print("=" * 60)
    X_train, X_test, y_train, y_test, pipeline, features = build_pipeline()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    r1, m1, p1              = model1_ols_full(X_train, X_test, y_train, y_test)
    r2, m2, p2, sel_feats   = model2_ols_selected(X_train, X_test, y_train, y_test)
    r3a, r3b, ridge, lasso, p3a, p3b = model3_ridge_lasso(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("ADVANCED METHODS")
    print("=" * 60)
    r4, poly_model                         = model4_polynomial(X_train, X_test, y_train, y_test, degree=2)
    r5, bayes_model, y_pred_bayes, y_std_b = model5b_bayesian_ridge(X_train, X_test, y_train, y_test)

    compare_models_table([r1, r2, r3a, r3b, r4, r5])
