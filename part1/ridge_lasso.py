import numpy as np

def ridge_fit(X, y, lam):
    """
    Cài đặt Ridge Regression.
    """
    p = X.shape[1]
    I_mod = np.eye(p)
    I_mod[0, 0] = 0  

    beta_hat = np.linalg.inv(X.T @ X + lam * I_mod) @ X.T @ y
    return beta_hat
