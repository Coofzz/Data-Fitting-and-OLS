import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


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


def ridge_fit(X, y, lam):
    """
    Cài đặt Ridge Regression.
    """
    p = X.shape[1]
    I_mod = np.eye(p)
    I_mod[0, 0] = 0  

    beta_hat = np.linalg.inv(X.T @ X + lam * I_mod) @ X.T @ y
    return beta_hat


def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư chuẩn theo chuẩn R lm():
      1. Residuals vs Fitted
      2. Normal Q-Q
      3. Scale-Location
      4. Cook's Distance
    """
    y_hat = X @ beta_hat
    residuals = y - y_hat
    n, p = X.shape

    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diagonal(H)

    sigma_hat = np.sqrt(np.sum(residuals ** 2) / (n - p))
    denom = sigma_hat * np.sqrt(np.clip(1 - leverage, 1e-10, None))
    std_residuals = residuals / denom

    cooks_d = (std_residuals ** 2 / p) * (leverage / np.clip(1 - leverage, 1e-10, None))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis Plots', fontsize=16)

    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='k', linewidths=0.4)
    z1 = np.polyfit(y_hat, residuals, 1)
    p1 = np.poly1d(z1)
    x_line = np.linspace(y_hat.min(), y_hat.max(), 200)
    axs[0, 0].plot(x_line, p1(x_line), color='red', lw=1.5)
    axs[0, 0].axhline(0, color='grey', linestyle='dashed', lw=1)
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')

    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')

    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))
    axs[1, 0].scatter(y_hat, sqrt_abs_std_res, alpha=0.6, edgecolors='k', linewidths=0.4)
    z3 = np.polyfit(y_hat, sqrt_abs_std_res, 1)
    p3 = np.poly1d(z3)
    axs[1, 0].plot(x_line, p3(x_line), color='red', lw=1.5)
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')

    obs_idx = np.arange(n)
    axs[1, 1].bar(obs_idx, cooks_d, color='steelblue', alpha=0.7)
    threshold = 4 / n         
    axs[1, 1].axhline(threshold, color='red', linestyle='dashed', lw=1.5,
                      label=f'Ngưỡng 4/n = {threshold:.3f}')
    influential = np.where(cooks_d > threshold)[0]
    for idx in influential:
        axs[1, 1].text(idx, cooks_d[idx] + cooks_d.max() * 0.01,
                       str(idx), ha='center', va='bottom', fontsize=7, color='red')
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation index')
    axs[1, 1].set_ylabel("Cook's Distance")
    axs[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def kfold_cv(X, y, k=5, random_state=42):
    """
    Cài đặt k-fold cross-validation cho mô hình OLS.
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
