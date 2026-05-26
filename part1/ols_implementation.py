import math

# Create a zero matrix
def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


# Compute the transpose of a matrix
def mat_transpose(A):
    rows, cols = len(A), len(A[0])
    AT = mat_zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            AT[j][i] = A[i][j]
    return AT

# Multiply two matrices
def mat_mul(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    assert len(B) == n, "Dimension mismatch for matrix multiplication."
    C = mat_zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Multiply a matrix by a vector
def mat_vec_mul(A, v):
    m, n = len(A), len(A[0])
    assert len(v) == n
    result = [0.0] * m
    for i in range(m):
        for j in range(n):
            result[i] += A[i][j] * v[j]
    return result

# Compute the inverse of a matrix
def mat_inverse(A):
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError(
                f"Singular matrix at column {col}. "
                "X^T X is not invertible."
            )
        aug[col] = [x / pivot for x in aug[col]]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][k] - factor * aug[col][k]
                        for k in range(2 * n)]

    return [aug[i][n:] for i in range(n)]


# Prepend a column of ones to X (intercept term)
def add_intercept(X):
    return [[1.0] + row for row in X]

# Function 1: ols_fit
def ols_fit(X, y, fit_intercept=True):
    X_design = add_intercept(X) if fit_intercept else [row[:] for row in X]
    n = len(X_design)          
    k = len(X_design[0])       
    p = k - 1 if fit_intercept else k  

    XT = mat_transpose(X_design)         

    XTX = mat_mul(XT, X_design)         

    XTX_inv = mat_inverse(XTX)           

    y_col = [[yi] for yi in y]           
    XTy   = mat_mul(XT, y_col)           
    XTy_vec = [row[0] for row in XTy]   

    beta_hat = mat_vec_mul(XTX_inv, XTy_vec) 

    y_hat = mat_vec_mul(X_design, beta_hat)   

    residuals = [y[i] - y_hat[i] for i in range(n)]

    rss = sum(r ** 2 for r in residuals)
    dof = n - k
    sigma2 = rss / (n - p - 1)

    return {
        "beta_hat" : beta_hat,
        "sigma2"   : sigma2,
        "rss"      : rss,
        "y_hat"    : y_hat,
        "residuals": residuals,
        "n"        : n,
        "p"        : p,
        "dof"      : dof,
    }

def print_results(result, feature_names=None):
    print("---------------------Function 1---------------------")
    p = result["p"]
    betas = result["beta_hat"]

    names = ["intercept"] + (
        feature_names if feature_names
        else [f"x{i}" for i in range(1, p + 1)]
    )

    print("  beta_hat:")
    for name, b in zip(names, betas):
        print(f"    {name:>12s} = {b:+.6f}")
    print(f"  sigma^2            = {result['sigma2']:.6f}")

# FUNCTION 2: HAT MATRIX
def mat_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
 
 
def hat_matrix(X, fit_intercept=True):
    X_design = add_intercept(X) if fit_intercept else [row[:] for row in X]
    n = len(X_design)
    k = len(X_design[0])
 
    XT = mat_transpose(X_design)                 
 
    XTX = mat_mul(XT, X_design)                   
 
    XTX_inv = mat_inverse(XTX)                    

    X_XTX_inv = mat_mul(X_design, XTX_inv)       
 
    H = mat_mul(X_XTX_inv, XT)                   
 
    tol = 1e-8
    H2 = mat_mul(H, H)
    is_idempotent = all(
        abs(H2[i][j] - H[i][j]) < tol
        for i in range(n) for j in range(n)
    )
 
    return {
        "H"            : H,
        "is_idempotent": is_idempotent,
        "n"            : n,
        "k"            : k,
    }
 
 
def print_hat_matrix(result):
    print("---------------------Function 2---------------------")
    H = result["H"]
 
    print("  Hat Matrix H =")
    for row in H:
        formatted = "  ".join(f"{v:+.4f}" for v in row)
        print(f"    [ {formatted} ]")
    print()
    print(f"Check H^2 = H : {'H satisfies the idempotent property' if result['is_idempotent'] else 'H does not satisfy the idempotent property'}")

# FUNCTION 3: MODEL METRICS
def model_metrics(y, y_hat, p, n=None):
    if n is None:
        n = len(y)
 
    
    rss = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
 
    y_mean = sum(y) / n
    tss = sum((yi - y_mean) ** 2 for yi in y)
 
    r2 = 1 - rss / tss
 
    df_resid = n - p - 1   
    df_model = p           
    r2_adj = 1 - (n - 1) / df_resid * (1 - r2)
 
    mss = (tss - rss) / df_model    
    mse = rss / df_resid           
    f_stat = mss / mse if mse > 1e-12 else float("inf")
 
 
    return {
        "rss"      : rss,
        "tss"      : tss,
        "r2"       : r2,
        "r2_adj"   : r2_adj,
        "f_stat"   : f_stat,
        "df_model" : df_model,
        "df_resid" : df_resid,
        "y_mean"   : y_mean,
    }
 
 
def print_metrics(result):
    print("---------------------Function 3---------------------")
    print(f"  RSS                : {result['rss']:.6f}")
    print(f"  TSS                : {result['tss']:.6f}")
    print(f"  R²                 : {result['r2']:.6f}")
    print(f"  Adjusted R²        : {result['r2_adj']:.6f}")
    print(f"  F-statistic        : {result['f_stat']:.6f}")


# ── Pure-Python approximation of the t-distribution (no scipy) ─────
def _regularized_incomplete_beta(x, a, b, max_iter=200, tol=1e-12):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1 - x, b, a, max_iter, tol)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f = 1.0; C = 1.0
    D = 1.0 - (a + b) * x / (a + 1)
    if abs(D) < 1e-30: D = 1e-30
    D = 1.0 / D; f = D
    for m in range(1, max_iter + 1):
        num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        D = 1 + num * D; C = 1 + num / C
        if abs(D) < 1e-30: D = 1e-30
        if abs(C) < 1e-30: C = 1e-30
        D = 1 / D; f *= C * D
        num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        D = 1 + num * D; C = 1 + num / C
        if abs(D) < 1e-30: D = 1e-30
        if abs(C) < 1e-30: C = 1e-30
        D = 1 / D; delta = C * D; f *= delta
        if abs(delta - 1) < tol: break
    return front * f


def _t_cdf(t_val, df):
    x = df / (df + t_val ** 2)
    p = 0.5 * _regularized_incomplete_beta(x, df / 2, 0.5)
    return p if t_val < 0 else 1 - p


def _t_sf(t_val, df):
    return 2 * (1 - _t_cdf(abs(t_val), df))


def _t_ppf(alpha, df, tol=1e-10, max_iter=200):
    if alpha < 0.5:
        return -_t_ppf(1 - alpha, df, tol, max_iter)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < alpha:
        hi *= 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < alpha: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return (lo + hi) / 2


# FUNCTION 4: coef_inference
def coef_inference(X, y, beta_hat, sigma2, fit_intercept=True, confidence=0.95):
    X_design = add_intercept(X) if fit_intercept else [row[:] for row in X]
    n = len(X_design)
    k = len(X_design[0])       
    dof = n - k                

    XT      = mat_transpose(X_design)
    XTX     = mat_mul(XT, X_design)
    XTX_inv = mat_inverse(XTX)

    se = [math.sqrt(sigma2 * XTX_inv[j][j]) for j in range(k)]

    t_stats = [beta_hat[j] / se[j] if se[j] > 1e-12 else float("inf")
               for j in range(k)]

    p_values = [_t_sf(t_stats[j], dof) for j in range(k)]

    alpha  = 1 - confidence
    t_crit = _t_ppf(1 - alpha / 2, dof)

    ci_lower = [beta_hat[j] - t_crit * se[j] for j in range(k)]
    ci_upper = [beta_hat[j] + t_crit * se[j] for j in range(k)]

    return {
        "se"      : se,
        "t_stats" : t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "dof"     : dof,
        "alpha"   : alpha,
        "t_crit"  : t_crit,
    }


def print_inference(result, feature_names=None, beta_hat=None):
    print("---------------------Function 4---------------------")
    k     = len(result["se"])
    names = ["intercept"] + (
        feature_names if feature_names
        else [f"x{i}" for i in range(1, k)]
    )
    print()
    print(f"  {'beta':>10s}  {'SE':>10s}  {'t-stat':>10s}  {'p-value':>10s}")
    print("  " + "-" * 50)
    for j in range(k):
        b = beta_hat[j] if beta_hat else float("nan")
        print(f"  {b:>10.4f}  {result['se'][j]:>10.4f}"
              f"  {result['t_stats'][j]:>10.4f}  {result['p_values'][j]:>10.4f}")

if __name__ == "__main__":
    print("Test 1: y = -1.5 + 1.5*x ")
    X1 = [
        [1.0],
        [3.0],
        [4.0],
        [7.0],
        [9.0],
        [12.0],
    ]
    y1 = [0.0, 2.0, 5.0, 10.0, 12.0, 16.0]
    
    result = ols_fit(X1, y1)
    print_results(result, feature_names=["x"])

    r_hat = hat_matrix(X1)
    print_hat_matrix(r_hat)

    metrics1 = model_metrics(y1, result["y_hat"], result["p"])
    print_metrics(metrics1)

    inf1 = coef_inference(X1, y1, result["beta_hat"], result["sigma2"])
    print_inference(inf1, feature_names=["x"], beta_hat=result["beta_hat"])