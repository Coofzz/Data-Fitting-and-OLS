import numpy as np


def kfold_cv(X, y, k=5, random_state=42):
    """
    K-fold cross-validation for OLS regression.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    mse_scores = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = np.hstack(folds[:i] + folds[i + 1:])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        y_pred = X_test @ beta
        mse = np.mean((y_test - y_pred) ** 2)
        mse_scores.append(mse)

    return np.mean(mse_scores), mse_scores
