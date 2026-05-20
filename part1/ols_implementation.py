import numpy as np

def vif(X):
    """
    Tính chỉ số VIF cho từng biến trong ma trận đặc trưng X.
    """
    n, p = X.shape
    vif_scores = np.zeros(p)

    for i in range(p):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)

        X_i = np.column_stack([np.ones(n), X_i])

        beta = np.linalg.inv(X_i.T @ X_i) @ X_i.T @ y_i

        y_hat = X_i @ beta

        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        ss_res = np.sum((y_i - y_hat) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared == 1:
            vif_scores[i] = np.inf
        else:
            vif_scores[i] = 1 / (1 - r_squared)

    return vif_scores
