import io
import math
import sys
import unittest

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ols_implementation import (
    ols_fit,
    hat_matrix,
    model_metrics,
    coef_inference,
    print_results,
    print_hat_matrix,
    print_metrics,
    print_inference,
)

TOL = 1e-4


def approx(a, b, tol=TOL):
    return abs(a - b) < tol


def _design_matrix(X, fit_intercept=True):
    X_np = np.asarray(X, dtype=float)
    if fit_intercept:
        return np.hstack([np.ones((len(X_np), 1)), X_np])
    return X_np


def numpy_ols_reference(X, y, fit_intercept=True):
    Xd = _design_matrix(X, fit_intercept)
    y_np = np.asarray(y, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(Xd, y_np, rcond=None)
    y_hat = Xd @ beta
    rss = float(np.sum((y_np - y_hat) ** 2))
    n, k = Xd.shape
    p = k - 1 if fit_intercept else k
    sigma2 = rss / (n - p - 1)
    return beta, y_hat, rss, sigma2


def numpy_hat_reference(X, fit_intercept=True):
    Xd = _design_matrix(X, fit_intercept)
    return Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T


def numpy_metrics_reference(y, y_hat, p, n=None):
    y_np = np.asarray(y, dtype=float)
    yh = np.asarray(y_hat, dtype=float)
    if n is None:
        n = len(y_np)
    rss = float(np.sum((y_np - yh) ** 2))
    tss = float(np.sum((y_np - y_np.mean()) ** 2))
    r2 = 1.0 - rss / tss
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)
    if rss < 1e-12:
        f_stat = float("inf")
    else:
        f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    return rss, tss, r2, r2_adj, f_stat


def _match_f_stat(a, b, tol=TOL):
    if math.isinf(a) and math.isinf(b):
        return True
    return approx(a, b, tol=tol)


def numpy_inference_reference(
    X, y, beta_hat, sigma2, fit_intercept=True, confidence=0.95
):
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for inference cross-check")
    Xd = _design_matrix(X, fit_intercept)
    n, k = Xd.shape
    dof = n - k
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    beta = np.asarray(beta_hat, dtype=float)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t_stats = beta / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), dof))
    t_crit = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, dof)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se
    return se, t_stats, p_values, ci_lower, ci_upper


def numpy_se_reference(X, sigma2, fit_intercept=True):
    Xd = _design_matrix(X, fit_intercept)
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    return np.sqrt(sigma2 * np.diag(xtx_inv))


def verify_against_numpy(X, y, fit_intercept=True, confidence=0.95, label=""):
    print(f"\n--- Verify against NumPy{': ' + label if label else ''} ---")
    result = ols_fit(X, y, fit_intercept=fit_intercept)
    beta_np, yhat_np, rss_np, s2_np = numpy_ols_reference(X, y, fit_intercept)

    checks = [
        ("ols_fit beta_hat", np.allclose(result["beta_hat"], beta_np, atol=TOL)),
        ("ols_fit y_hat", np.allclose(result["y_hat"], yhat_np, atol=TOL)),
        ("ols_fit rss", approx(result["rss"], rss_np, tol=TOL)),
        ("ols_fit sigma2", approx(result["sigma2"], s2_np, tol=TOL)),
    ]

    H_ours = np.array(hat_matrix(X, fit_intercept=fit_intercept)["H"])
    H_np = numpy_hat_reference(X, fit_intercept)
    checks += [
        ("hat_matrix H", np.allclose(H_ours, H_np, atol=TOL)),
        ("hat_matrix H^2=H", np.allclose(H_ours @ H_ours, H_ours, atol=TOL)),
    ]

    metrics = model_metrics(y, result["y_hat"], result["p"])
    rss_m, tss_m, r2_m, r2a_m, f_m = numpy_metrics_reference(
        y, result["y_hat"], result["p"]
    )
    checks += [
        ("model_metrics rss", approx(metrics["rss"], rss_m, tol=TOL)),
        ("model_metrics r2", approx(metrics["r2"], r2_m, tol=TOL)),
        ("model_metrics r2_adj", approx(metrics["r2_adj"], r2a_m, tol=TOL)),
        ("model_metrics f_stat", _match_f_stat(metrics["f_stat"], f_m)),
    ]

    inf = coef_inference(
        X, y, result["beta_hat"], result["sigma2"],
        fit_intercept=fit_intercept, confidence=confidence,
    )
    se_np = numpy_se_reference(X, result["sigma2"], fit_intercept)
    checks.append(("coef_inference se", np.allclose(inf["se"], se_np, atol=TOL)))

    if HAS_SCIPY:
        _, t_np, pv_np, lo_np, hi_np = numpy_inference_reference(
            X, y, result["beta_hat"], result["sigma2"],
            fit_intercept=fit_intercept, confidence=confidence,
        )
        checks.append(
            ("coef_inference p_values", np.allclose(inf["p_values"], pv_np, atol=TOL))
        )
        if result["sigma2"] > 1e-8:
            checks += [
                ("coef_inference t_stats", np.allclose(
                    inf["t_stats"], t_np, atol=TOL, rtol=TOL
                )),
                ("coef_inference ci_lower", np.allclose(inf["ci_lower"], lo_np, atol=TOL)),
                ("coef_inference ci_upper", np.allclose(inf["ci_upper"], hi_np, atol=TOL)),
            ]
        else:
            print("  [SKIP]  coef_inference t/CI (sigma^2 ~ 0, perfect fit)")
    else:
        print("  [SKIP]  coef_inference t/p/CI (install scipy)")

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    return all(ok for _, ok in checks)


def run_pipeline(test_name, X, y, feature_names):
    print(f"\n{test_name}")
    result = ols_fit(X, y)
    print_results(result, feature_names=feature_names)

    r_hat = hat_matrix(X)
    print_hat_matrix(r_hat)

    metrics = model_metrics(y, result["y_hat"], result["p"])
    print_metrics(metrics)

    inf = coef_inference(X, y, result["beta_hat"], result["sigma2"])
    print_inference(inf, feature_names=feature_names, beta_hat=result["beta_hat"])

    return result, r_hat, metrics, inf


class TestOLS(unittest.TestCase):

    def test_1_linear_regression(self):
        X = [
            [1.0],
            [3.0],
            [4.0],
            [7.0],
            [9.0],
            [12.0],
        ]
        y = [0.0, 2.0, 5.0, 10.0, 12.0, 16.0]

        result, r_hat, metrics, inf = run_pipeline(
            "Test 1: y = -1.5 + 1.5*x",
            X,
            y,
            feature_names=["x"],
        )

        self.assertTrue(approx(result["beta_hat"][0], -1.5))
        self.assertTrue(approx(result["beta_hat"][1], 1.5))
        self.assertTrue(approx(sum(result["residuals"]), 0.0, tol=1e-6))
        self.assertTrue(r_hat["is_idempotent"])
        self.assertEqual(len(r_hat["H"]), len(X))
        self.assertTrue(0.0 <= metrics["r2"] <= 1.0)
        self.assertGreater(metrics["f_stat"], 0.0)
        for se in inf["se"]:
            self.assertGreater(se, 0.0)
        for pv in inf["p_values"]:
            self.assertTrue(0.0 <= pv <= 1.0)
        self.assertTrue(verify_against_numpy(X, y, label="Test 1: y = -1.5 + 1.5*x"))

    def test_2_perfect_linear_fit(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [5.0, 8.0, 11.0, 14.0, 17.0]

        result, r_hat, metrics, inf = run_pipeline(
            "Test 2: y = 2 + 3*x",
            X,
            y,
            feature_names=["x"],
        )

        self.assertTrue(approx(result["beta_hat"][0], 2.0))
        self.assertTrue(approx(result["beta_hat"][1], 3.0))
        self.assertTrue(approx(result["rss"], 0.0, tol=1e-6))
        self.assertTrue(approx(metrics["r2"], 1.0))
        self.assertTrue(r_hat["is_idempotent"])
        trace = sum(r_hat["H"][i][i] for i in range(r_hat["n"]))
        self.assertTrue(approx(trace, r_hat["k"]))
        self.assertLess(inf["p_values"][1], 0.05)
        self.assertTrue(verify_against_numpy(X, y, label="Test 2: y = 2 + 3*x"))

    def test_3_no_intercept(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [2.0, 4.0, 6.0, 8.0]

        print("\nTest 3: y = 2*x (no intercept)")
        result = ols_fit(X, y, fit_intercept=False)
        print_results(result, feature_names=["x"])

        r_hat = hat_matrix(X, fit_intercept=False)
        print_hat_matrix(r_hat)

        metrics = model_metrics(y, result["y_hat"], result["p"])
        print_metrics(metrics)

        inf = coef_inference(
            X, y, result["beta_hat"], result["sigma2"], fit_intercept=False
        )
        print_inference(inf, feature_names=["x"], beta_hat=result["beta_hat"])

        self.assertEqual(len(result["beta_hat"]), 1)
        self.assertTrue(approx(result["beta_hat"][0], 2.0))
        self.assertTrue(approx(result["rss"], 0.0, tol=1e-6))
        self.assertTrue(verify_against_numpy(X, y, fit_intercept=False, label="Test 3: y = 2*x (no intercept)"))

    def test_4_multivariate_regression(self):
        X = [
            [1.0, 0.5],
            [2.0, 3.0],
            [3.0, 1.5],
            [4.0, 4.0],
            [5.0, 2.5],
            [6.0, 1.0],
        ]
        y = [1.0 + 2.0 * x1 + 0.5 * x2 for x1, x2 in X]

        result, r_hat, metrics, inf = run_pipeline(
            "Test 4: y = 1 + 2*x1 + 0.5*x2",
            X,
            y,
            feature_names=["x1", "x2"],
        )

        self.assertEqual(len(result["beta_hat"]), 3)
        self.assertTrue(approx(result["beta_hat"][0], 1.0))
        self.assertTrue(approx(result["beta_hat"][1], 2.0))
        self.assertTrue(approx(result["beta_hat"][2], 0.5))
        self.assertTrue(approx(result["rss"], 0.0, tol=1e-6))
        self.assertTrue(approx(metrics["r2"], 1.0))
        self.assertTrue(r_hat["is_idempotent"])
        n = len(r_hat["H"])
        for i in range(n):
            for j in range(n):
                self.assertTrue(approx(r_hat["H"][i][j], r_hat["H"][j][i]))
        for j in range(3):
            lo = min(inf["ci_lower"][j], inf["ci_upper"][j])
            hi = max(inf["ci_lower"][j], inf["ci_upper"][j])
            self.assertLessEqual(lo, result["beta_hat"][j] + 1e-8)
            self.assertGreaterEqual(hi, result["beta_hat"][j] - 1e-8)
        self.assertTrue(verify_against_numpy(X, y, label="Test 4: y = 1 + 2*x1 + 0.5*x2"))

    def test_5_wider_ci_at_99_percent(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        y = [2.1, 4.0, 5.9, 8.1, 9.9, 12.0]

        result, r_hat, metrics, inf = run_pipeline(
            "Test 5: y ~ 2*x ",
            X,
            y,
            feature_names=["x"],
        )

        inf95 = coef_inference(
            X, y, result["beta_hat"], result["sigma2"], confidence=0.95
        )
        inf99 = coef_inference(
            X, y, result["beta_hat"], result["sigma2"], confidence=0.99
        )

        self.assertLessEqual(metrics["r2_adj"], metrics["r2"] + 1e-8)
        self.assertLessEqual(metrics["rss"], metrics["tss"] + 1e-8)
        for j in range(len(result["beta_hat"])):
            width95 = abs(inf95["ci_upper"][j] - inf95["ci_lower"][j])
            width99 = abs(inf99["ci_upper"][j] - inf99["ci_lower"][j])
            self.assertGreater(width99, width95)
        self.assertLess(inf["p_values"][1], 0.05)
        self.assertTrue(verify_against_numpy(X, y, label="Test 5: y ~ 2*x"))


@unittest.skipUnless(HAS_SCIPY, "pip install scipy to verify p-value/CI")
class TestNumpyInferenceSciPy(unittest.TestCase):

    def test_inference_scipy_test1(self):
        X = [[1.0], [3.0], [4.0], [7.0], [9.0], [12.0]]
        y = [0.0, 2.0, 5.0, 10.0, 12.0, 16.0]
        result = ols_fit(X, y)
        inf = coef_inference(X, y, result["beta_hat"], result["sigma2"])
        _, _, pv_np, lo_np, hi_np = numpy_inference_reference(
            X, y, result["beta_hat"], result["sigma2"]
        )
        self.assertTrue(np.allclose(inf["p_values"], pv_np, atol=TOL))
        self.assertTrue(np.allclose(inf["ci_lower"], lo_np, atol=TOL))
        self.assertTrue(np.allclose(inf["ci_upper"], hi_np, atol=TOL))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestOLS))
    suite.addTests(loader.loadTestsFromTestCase(TestNumpyInferenceSciPy))
    runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
    result = runner.run(suite)

    if not result.wasSuccessful():
        for test, err in result.failures + result.errors:
            print(f"\nFAILED: {test}")
            print(err)
    sys.exit(0 if result.wasSuccessful() else 1)