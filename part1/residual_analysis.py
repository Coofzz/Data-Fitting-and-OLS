import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from ols_implementation import mat_transpose, mat_mul, mat_inverse, mat_vec_mul

def simple_ols(x, y):
    n = len(x)
    x_mean = sum(x)/n
    y_mean = sum(y)/n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean)**2 for i in range(n))
    slope = num / den if den > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    return intercept, slope

def residual_plots(X, y, beta_hat):
    """
    Draw the four standard residual diagnostic plots.
    """
    X = [list(row) for row in X]
    y = list(y)
    beta_hat = list(beta_hat)
    n = len(X)
    p = len(X[0])
    
    y_hat = mat_vec_mul(X, beta_hat)
    residuals = [y[i] - y_hat[i] for i in range(n)]
    
    XT = mat_transpose(X)
    XTX = mat_mul(XT, X)
    XTX_inv = mat_inverse(XTX)
    X_XTX_inv = mat_mul(X, XTX_inv)
    H = mat_mul(X_XTX_inv, XT)
    leverage = [H[i][i] for i in range(n)]
    
    rss = sum(r**2 for r in residuals)
    sigma_hat = math.sqrt(rss / (n - p))
    
    std_residuals = []
    cooks_d = []
    for i in range(n):
        lev = max(1 - leverage[i], 1e-10)
        denom = sigma_hat * math.sqrt(lev)
        sr = residuals[i] / denom if denom > 1e-12 else 0.0
        std_residuals.append(sr)
        
        cd = (sr**2 / p) * (leverage[i] / lev)
        cooks_d.append(cd)
        
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Diagnostic Plots', fontsize=16)
    
    y_hat_min = min(y_hat)
    y_hat_max = max(y_hat)
    step = (y_hat_max - y_hat_min) / 199 if y_hat_max > y_hat_min else 1
    x_line = [y_hat_min + i * step for i in range(200)]
    
    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='k', linewidths=0.4)
    int1, slope1 = simple_ols(y_hat, residuals)
    trend1_y = [int1 + slope1 * x for x in x_line]
    axs[0, 0].plot(x_line, trend1_y, color='red', lw=1.5, label='Trend')
    axs[0, 0].axhline(0, color='grey', linestyle='dashed', lw=1)
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].legend(fontsize=9)
    
    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')
    
    sqrt_abs_std = [math.sqrt(abs(sr)) for sr in std_residuals]
    axs[1, 0].scatter(y_hat, sqrt_abs_std, alpha=0.6, edgecolors='k', linewidths=0.4)
    int3, slope3 = simple_ols(y_hat, sqrt_abs_std)
    trend3_y = [int3 + slope3 * x for x in x_line]
    axs[1, 0].plot(x_line, trend3_y, color='red', lw=1.5, label='Trend')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
    axs[1, 0].legend(fontsize=9)
    
    threshold = 4 / n
    obs_idx = list(range(n))
    axs[1, 1].bar(obs_idx, cooks_d, color='steelblue', alpha=0.7)
    axs[1, 1].axhline(threshold, color='red', linestyle='dashed', lw=1.5, label=f"Threshold 4/n = {threshold:.3f}")
    
    cooks_d_max = max(cooks_d)
    for i in range(n):
        if cooks_d[i] > threshold:
            axs[1, 1].text(i, cooks_d[i] + cooks_d_max * 0.01, str(i), ha='center', va='bottom', fontsize=7, color='red')
            
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation index')
    axs[1, 1].set_ylabel("Cook's Distance")
    axs[1, 1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
