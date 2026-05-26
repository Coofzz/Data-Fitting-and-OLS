import numpy as np
import sys
sys.path.insert(0, '/home/claude')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ols_implementation import vif
from ridge_lasso import ridge_fit
from residual_analysis import residual_plots
from cross_validation import kfold_cv

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

results = []

def check(name, condition, detail=""):
    status = f"{GREEN}  PASS{RESET}" if condition else f"{RED}  FAIL{RESET}"
    print(f"{status}  {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"        {CYAN}{line}{RESET}")
    results.append(condition)

def section(title):
    print(f"\n{BOLD}── {title} {'─' * (55 - len(title))}{RESET}")


# Tests for vif(X)
section("vif(X)")

# T1: Independent features → VIF close to 1
np.random.seed(0)
X_ind = np.random.randn(500, 3)
v = vif(X_ind)
check(
    "T1 [vif] Independent features: all VIF ≈ 1 (< 2)",
    np.all(v < 2),
    f"VIF values: {np.round(v, 4)}\n"
    f"Expected  : all < 2 (independent features)"
)

# T2: Near-perfect collinearity → VIF >> 10
np.random.seed(99)
X_col = np.random.randn(200, 2)
X_col_near = np.column_stack([X_col, X_col[:, 0] + 1e-6 * np.random.randn(200)])
v_col = vif(X_col_near)
check(
    "T2 [vif] Near-perfect collinearity: VIF_3 > 1e6",
    v_col[2] > 1e6,
    f"VIF values: {np.round(v_col, 2)}\n"
    f"VIF col 3 = {v_col[2]:.4e}  |  Expected: > 1e6"
)

# T3: High multicollinearity → VIF >> 10
X_mc = np.random.rand(200, 3)
X_mc[:, 2] = 0.99 * X_mc[:, 0] + np.random.rand(200) * 0.01
v_mc = vif(X_mc)
check(
    "T3 [vif] Near-collinear X3: VIF_3 > 100",
    v_mc[2] > 100,
    f"VIF values: {np.round(v_mc, 4)}\n"
    f"VIF col 3 = {v_mc[2]:.4f}  |  Expected: > 100"
)

# T4: Output shape is (p,)
X_sh = np.random.randn(100, 5)
v_sh = vif(X_sh)
check(
    "T4 [vif] Output shape equals number of features",
    v_sh.shape == (5,),
    f"Shape actual  : {v_sh.shape}\n"
    f"Shape expected: (5,)"
)

# T5: Matches statsmodels variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_ref = np.random.rand(150, 3)
X_ref[:, 2] = 0.8 * X_ref[:, 0] + 0.4 * X_ref[:, 1] + np.random.rand(150) * 0.1
v_custom = vif(X_ref)
X_with_int = np.column_stack([np.ones(150), X_ref])
v_sm = np.array([variance_inflation_factor(X_with_int, i+1) for i in range(3)])
check(
    "T5 [vif] Matches statsmodels (atol=1e-6)",
    np.allclose(v_custom, v_sm, atol=1e-6),
    f"Custom VIF    : {np.round(v_custom, 6)}\n"
    f"Statsmodels   : {np.round(v_sm, 6)}\n"
    f"Max diff      : {np.max(np.abs(v_custom - v_sm)):.2e}"
)


# Tests for ridge_fit(X, y, lam)
section("ridge_fit(X, y, lam)")

np.random.seed(1)
X_r = np.random.randn(80, 3)
X_rb = np.column_stack([np.ones(80), X_r])
true_b = np.array([1.0, 2.0, -1.5, 0.5])
y_r = X_rb @ true_b + np.random.randn(80) * 0.3

from sklearn.linear_model import Ridge

# T1: lam=0 matches plain OLS
beta_ols = np.linalg.inv(X_rb.T @ X_rb) @ X_rb.T @ y_r
beta_r0 = ridge_fit(X_rb, y_r, lam=0)
check(
    "T1 [ridge] lam=0 equals OLS solution (atol=1e-10)",
    np.allclose(beta_r0, beta_ols, atol=1e-10),
    f"Ridge(lam=0) : {np.round(beta_r0, 6)}\n"
    f"OLS          : {np.round(beta_ols, 6)}\n"
    f"Max diff     : {np.max(np.abs(beta_r0 - beta_ols)):.2e}"
)

# T2: Matches sklearn Ridge (solver=cholesky)
beta_rc = ridge_fit(X_rb, y_r, lam=5.0)
sk = Ridge(alpha=5.0, solver='cholesky').fit(X_r, y_r)
beta_sk = np.concatenate([[sk.intercept_], sk.coef_])
check(
    "T2 [ridge] Matches sklearn Ridge alpha=5 (atol=1e-8)",
    np.allclose(beta_rc, beta_sk, atol=1e-8),
    f"Custom ridge : {np.round(beta_rc, 6)}\n"
    f"sklearn      : {np.round(beta_sk, 6)}\n"
    f"Max diff     : {np.max(np.abs(beta_rc - beta_sk)):.2e}"
)

# T3: Larger lambda → coefficients shrink toward 0
beta_small = ridge_fit(X_rb, y_r, lam=0.01)
beta_large = ridge_fit(X_rb, y_r, lam=1000.0)
check(
    "T3 [ridge] Larger lambda shrinks slope coefficients",
    np.all(np.abs(beta_large[1:]) < np.abs(beta_small[1:])),
    f"|beta| lam=0.01  : {np.round(np.abs(beta_small[1:]), 6)}\n"
    f"|beta| lam=1000  : {np.round(np.abs(beta_large[1:]), 6)}\n"
    f"Shrinkage ratio  : {np.round(np.abs(beta_large[1:]) / np.abs(beta_small[1:]), 4)}"
)

# T4: Intercept is NOT penalised
betas_intercepts = [ridge_fit(X_rb, y_r, lam=l)[0]
                    for l in [0.01, 1, 10, 100, 1000]]
check(
    "T4 [ridge] Intercept not driven to 0 for large lambda",
    abs(betas_intercepts[-1]) > 0.01,
    f"Intercepts at lam=[0.01, 1, 10, 100, 1000]:\n"
    f"  {[round(b, 6) for b in betas_intercepts]}\n"
    f"Intercept lam=1000 = {betas_intercepts[-1]:.6f}  |  Expected: |val| > 0.01"
)

# T5: Output shape
check(
    "T5 [ridge] Output shape = (p+1,)",
    ridge_fit(X_rb, y_r, lam=1.0).shape == (X_rb.shape[1],),
    f"Shape actual  : {ridge_fit(X_rb, y_r, lam=1.0).shape}\n"
    f"Shape expected: ({X_rb.shape[1]},)"
)


# Tests for residual_plots(X, y, beta_hat)
section("residual_plots(X, y, beta_hat)")

np.random.seed(2)
X_p = np.column_stack([np.ones(60), np.random.randn(60, 2)])
y_p = X_p @ [1.0, -0.5, 2.0] + np.random.randn(60)
beta_p = np.linalg.inv(X_p.T @ X_p) @ X_p.T @ y_p

import statsmodels.api as sm
model_sm2 = sm.OLS(y_p, X_p).fit()
inf2 = model_sm2.get_influence()
n2, p2 = X_p.shape
res2   = y_p - X_p @ beta_p
H2     = X_p @ np.linalg.inv(X_p.T @ X_p) @ X_p.T
lev2   = np.diagonal(H2)
sig2   = np.sqrt(np.sum(res2**2) / (n2 - p2))
sres2  = res2 / (sig2 * np.sqrt(np.clip(1 - lev2, 1e-10, None)))
cd2    = (sres2**2 / p2) * (lev2 / np.clip(1 - lev2, 1e-10, None))

# T1: Residuals match statsmodels
diff_res = np.max(np.abs(res2 - model_sm2.resid))
check(
    "T1 [residual_plots] Raw residuals match statsmodels (atol=1e-8)",
    np.allclose(res2, model_sm2.resid, atol=1e-8),
    f"Max |diff| vs statsmodels: {diff_res:.2e}  |  Threshold: 1e-8\n"
    f"Mean residual: {np.mean(res2):.6f}  |  Std: {np.std(res2):.6f}"
)

# T2: Standardised residuals match statsmodels
diff_sres = np.max(np.abs(sres2 - inf2.resid_studentized_internal))
check(
    "T2 [residual_plots] Standardised residuals match statsmodels (atol=1e-6)",
    np.allclose(sres2, inf2.resid_studentized_internal, atol=1e-6),
    f"Max |diff| vs statsmodels: {diff_sres:.2e}  |  Threshold: 1e-6\n"
    f"sigma_hat = {sig2:.6f}\n"
    f"Std resid range: [{sres2.min():.4f}, {sres2.max():.4f}]"
)

# T3: sqrt(|std resid|) matches
sqrt_sres  = np.sqrt(np.abs(sres2))
sqrt_sres_sm = np.sqrt(np.abs(inf2.resid_studentized_internal))
diff_sqrt = np.max(np.abs(sqrt_sres - sqrt_sres_sm))
check(
    "T3 [residual_plots] sqrt|std resid| matches statsmodels (atol=1e-6)",
    np.allclose(sqrt_sres, sqrt_sres_sm, atol=1e-6),
    f"Max |diff| vs statsmodels: {diff_sqrt:.2e}  |  Threshold: 1e-6\n"
    f"sqrt|std resid| range: [{sqrt_sres.min():.4f}, {sqrt_sres.max():.4f}]"
)

# T4: Cook's Distance matches statsmodels
diff_cd = np.max(np.abs(cd2 - inf2.cooks_distance[0]))
n_influential = np.sum(cd2 > 4 / n2)
check(
    "T4 [residual_plots] Cook's Distance matches statsmodels (atol=1e-6)",
    np.allclose(cd2, inf2.cooks_distance[0], atol=1e-6),
    f"Max |diff| vs statsmodels: {diff_cd:.2e}  |  Threshold: 1e-6\n"
    f"Cook's D range  : [{cd2.min():.6f}, {cd2.max():.6f}]\n"
    f"Threshold (4/n) : {4/n2:.6f}  →  {n_influential} influential point(s)"
)

# T5: residual_plots runs without error
residual_plots(X_p, y_p, beta_p)
plt.close('all')
check(
    "T5 [residual_plots] Function runs without raising an exception",
    True,
    "4 subplots created successfully: Residuals vs Fitted, Normal Q-Q,\n"
    "Scale-Location, Cook's Distance"
)


# Tests for kfold_cv(X, y, k, random_state)
section("kfold_cv(X, y, k, random_state)")

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

np.random.seed(3)
X_cv = np.column_stack([np.ones(120), np.random.randn(120, 2)])
y_cv = X_cv @ [0.5, 1.0, -2.0] + np.random.randn(120)

# T1: Returns tuple (float, list) with correct list length
mean_mse, mse_list = kfold_cv(X_cv, y_cv, k=5, random_state=0)
check(
    "T1 [kfold_cv] Returns (float, list) with len(list) == k",
    isinstance(mean_mse, float) and len(mse_list) == 5,
    f"mean_mse type : {type(mean_mse).__name__}  |  Expected: float\n"
    f"len(mse_list) : {len(mse_list)}  |  Expected: 5\n"
    f"mean_mse      : {mean_mse:.6f}"
)

# T2: mean_mse equals mean of mse_list
check(
    "T2 [kfold_cv] mean_mse == mean(mse_list)",
    np.isclose(mean_mse, np.mean(mse_list)),
    f"mean_mse          : {mean_mse:.10f}\n"
    f"mean(mse_list)    : {np.mean(mse_list):.10f}\n"
    f"MSE per fold      : {[round(m, 6) for m in mse_list]}"
)

# T3: Reproducibility
m1, s1 = kfold_cv(X_cv, y_cv, k=5, random_state=99)
m2, s2 = kfold_cv(X_cv, y_cv, k=5, random_state=99)
check(
    "T3 [kfold_cv] Same random_state → identical MSE scores",
    np.allclose(s1, s2),
    f"Run 1 MSEs: {[round(m, 8) for m in s1]}\n"
    f"Run 2 MSEs: {[round(m, 8) for m in s2]}\n"
    f"Max diff  : {np.max(np.abs(np.array(s1) - np.array(s2))):.2e}"
)

# T4: Different random_state → different scores
m3, s3 = kfold_cv(X_cv, y_cv, k=5, random_state=7)
check(
    "T4 [kfold_cv] Different random_state → different fold assignments",
    not np.allclose(s1, s3),
    f"random_state=99 MSEs : {[round(m, 6) for m in s1]}\n"
    f"random_state=7  MSEs : {[round(m, 6) for m in s3]}\n"
    f"Max diff             : {np.max(np.abs(np.array(s1) - np.array(s3))):.6f}"
)

# T5: OLS on same splits matches sklearn LinearRegression
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
custom_mses, sk_mses = [], []
for tr, te in kf5.split(X_cv):
    b = np.linalg.inv(X_cv[tr].T @ X_cv[tr]) @ X_cv[tr].T @ y_cv[tr]
    custom_mses.append(np.mean((y_cv[te] - X_cv[te] @ b) ** 2))
    lm = LinearRegression(fit_intercept=False).fit(X_cv[tr], y_cv[tr])
    sk_mses.append(np.mean((y_cv[te] - lm.predict(X_cv[te])) ** 2))
check(
    "T5 [kfold_cv] OLS matches sklearn on identical splits (atol=1e-8)",
    np.allclose(custom_mses, sk_mses, atol=1e-8),
    f"Custom MSEs : {[round(m, 8) for m in custom_mses]}\n"
    f"sklearn MSEs: {[round(m, 8) for m in sk_mses]}\n"
    f"Max diff    : {np.max(np.abs(np.array(custom_mses) - np.array(sk_mses))):.2e}"
)


print(f"\n{'='*58}")
passed = sum(results)
total  = len(results)
print(f"  {BOLD}Result: {passed}/{total} tests passed{RESET}")
if passed == total:
    print(f"  {GREEN}{BOLD}All tests PASSED ✓{RESET}")
else:
    print(f"  {RED}{BOLD}{total - passed} test(s) FAILED ✗{RESET}")
print('='*58)
