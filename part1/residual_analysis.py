import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats


def residual_plots(X, y, beta_hat):
    """
    Draw the four standard residual diagnostic plots (R lm() style):
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
    fig.suptitle('Residual Diagnostic Plots', fontsize=16)

    x_line = np.linspace(y_hat.min(), y_hat.max(), 200)
    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='k', linewidths=0.4)
    trend1 = np.poly1d(np.polyfit(y_hat, residuals, 1))
    axs[0, 0].plot(x_line, trend1(x_line), color='red', lw=1.5, label='Trend')
    axs[0, 0].axhline(0, color='grey', linestyle='dashed', lw=1)
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].legend(fontsize=9)

    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')

    sqrt_abs_std = np.sqrt(np.abs(std_residuals))
    axs[1, 0].scatter(y_hat, sqrt_abs_std, alpha=0.6, edgecolors='k', linewidths=0.4)
    trend3 = np.poly1d(np.polyfit(y_hat, sqrt_abs_std, 1))
    axs[1, 0].plot(x_line, trend3(x_line), color='red', lw=1.5, label='Trend')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
    axs[1, 0].legend(fontsize=9)

    threshold = 4 / n
    obs_idx = np.arange(n)
    axs[1, 1].bar(obs_idx, cooks_d, color='steelblue', alpha=0.7)
    axs[1, 1].axhline(threshold, color='red', linestyle='dashed', lw=1.5,
                      label=f"Threshold 4/n = {threshold:.3f}")
    for idx in np.where(cooks_d > threshold)[0]:
        axs[1, 1].text(idx, cooks_d[idx] + cooks_d.max() * 0.01,
                       str(idx), ha='center', va='bottom', fontsize=7, color='red')
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation index')
    axs[1, 1].set_ylabel("Cook's Distance")
    axs[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
