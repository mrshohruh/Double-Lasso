
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
import statsmodels.api as sm
from scipy.stats import norm

# -----------------------------
# Penalty, residualization, core DL-CI
# -----------------------------

def plugin_alpha(n, p, c=1.1):
    return c * np.sqrt(2 * np.log(p) / n)

def lasso_residuals(X, y, alpha):
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=3000)
    model.fit(X, y)
    y_hat = model.predict(X)
    return y - y_hat, model

def double_lasso_ci(Y, D, X, alpha=None, ci_level=0.95):
    n, p = X.shape
    if alpha is None:
        alpha = plugin_alpha(n, p)

    Y_resid, model_y = lasso_residuals(X, Y, alpha)
    D_resid, model_d = lasso_residuals(X, D, alpha)

    X_ols = sm.add_constant(D_resid)
    ols_fit = sm.OLS(Y_resid, X_ols).fit(cov_type="HC3")

    beta1_hat = ols_fit.params[1]
    se = ols_fit.bse[1]

    z = norm.ppf(0.975)
    ci_low, ci_high = beta1_hat - z * se, beta1_hat + z * se

    return {
        "beta1_hat": beta1_hat,
        "se_HC3": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "k_y": int(np.sum(model_y.coef_ != 0)),
        "k_d": int(np.sum(model_d.coef_ != 0)),
        "alpha": alpha
    }

# -----------------------------
# DGP
# -----------------------------

def simulate_dgp(n=200, p=100, s=5, beta1=2.0, rho=0.0, seed=None):
    rng = np.random.default_rng(seed)
    if rho == 0.0:
        X = rng.normal(size=(n, p))
    else:
        L = np.linalg.cholesky((1 - rho) * np.eye(p) + rho * np.ones((p, p)))
        X = rng.normal(size=(n, p)) @ L.T

    signal_x = X[:, :s].sum(axis=1)
    v = rng.normal(size=n)
    u = rng.normal(size=n)
    D = signal_x + v
    Y = beta1 * D + signal_x + u
    return Y, D, X

# -----------------------------
# Simulation loop
# -----------------------------

def run_simulation(R=100, n=200, p=100, s=5, beta1=2.0, rho=0.2, ci_level=0.95, c=1.1, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        Y, D, X = simulate_dgp(n, p, s, beta1, rho, seed=rng.integers(0, 1_000_000))
        alpha = plugin_alpha(n, p, c=c)
        est = double_lasso_ci(Y, D, X, alpha=alpha, ci_level=ci_level)
        covered = int((est["ci_low"] <= beta1) and (beta1 <= est["ci_high"]))
        rows.append({
            "beta1_hat": est["beta1_hat"],
            "se_HC3": est["se_HC3"],
            "ci_low": est["ci_low"],
            "ci_high": est["ci_high"],
            "ci_length": est["ci_high"] - est["ci_low"],
            "covered": covered,
            "k_y": est["k_y"],
            "k_d": est["k_d"],
            "alpha": est["alpha"]
        })
    return pd.DataFrame(rows)

# -----------------------------
# NEW: design sweep utility
# -----------------------------

def sweep_designs(param="n", values=(100, 200, 400), base=dict(n=200, p=100, s=5, beta1=2.0, rho=0.2),
                  R=100, ci_level=0.95, c=1.1, seed=123, make_plot=True, save_csv=None):
    """
    Sweep over a single design parameter and summarize coverage vs CI length.
    param: "n" | "p" | "s" | "rho"
    values: iterable of values for that parameter
    base: dict with baseline values for remaining parameters
    Returns a DataFrame with rows per design.
    """
    rows = []
    for val in values:
        kwargs = dict(base)
        kwargs[param] = val
        df = run_simulation(R=R, ci_level=ci_level, c=c, seed=seed, **kwargs)
        coverage = df["covered"].mean()
        avg_len = df["ci_length"].mean()
        bias = (df["beta1_hat"] - base.get("beta1", 2.0)).mean()
        rmse = np.sqrt(((df["beta1_hat"] - base.get("beta1", 2.0))**2).mean())
        rows.append({
            "param": param,
            "value": val,
            "coverage": coverage,
            "avg_ci_length": avg_len,
            "bias": bias,
            "rmse": rmse,
            "avg_k_y": df["k_y"].mean(),
            "avg_k_d": df["k_d"].mean()
        })
    out = pd.DataFrame(rows)
    if save_csv is not None:
        out.to_csv(save_csv, index=False)
    if make_plot:
        plt.figure()
        plt.plot(out["avg_ci_length"], out["coverage"], marker="o")
        for i, row in out.iterrows():
            plt.annotate(str(row["value"]), (row["avg_ci_length"], row["coverage"]), textcoords="offset points", xytext=(5,5), fontsize=9)
        plt.xlabel("Average CI length")
        plt.ylabel("Coverage")
        plt.title(f"Coverage vs CI length (sweep {param})")
        plt.grid(True, alpha=0.3)
        plt.show()
    return out

if __name__ == "__main__":
    # Quick demo run
    df = run_simulation(R=60, n=200, p=100, s=5, beta1=2.0, rho=0.2, seed=123)
    print(df.describe())

    # Sweep over n
    sweep = sweep_designs(param="n", values=(80, 120, 200, 320), R=60, save_csv="sweep_results.csv")
    print(sweep)
