
import numpy as np
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from scipy.stats import norm

def plugin_alpha(n, p, c=1.1):
    """Plug-in style L1 penalty for sklearn's Lasso ((1/2n)||y-Xb||^2 + alpha||b||_1)."""
    return c * np.sqrt(2 * np.log(p) / n)

def lasso_residuals(X, y, alpha):
    """Fit Lasso(y ~ X) and return residuals y - X b_hat - intercept."""
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=3000)
    model.fit(X, y)
    y_hat = model.predict(X)
    return y - y_hat, model

def double_lasso_ci(Y, D, X, alpha=None, ci_level=0.95):
    """
    Double LASSO:
      1) Lasso Y ~ X -> residuals Y~.
      2) Lasso D ~ X -> residuals D~.
      3) OLS Y~ on D~, HC3 robust SE -> CI for beta1.
    """
    n, p = X.shape
    if alpha is None:
        alpha = plugin_alpha(n, p)

    # Step 1: partial out X from Y
    Y_resid, model_y = lasso_residuals(X, Y, alpha)
    # Step 2: partial out X from D
    D_resid, model_d = lasso_residuals(X, D, alpha)

    # Step 3: OLS on residuals with HC3
    X_ols = sm.add_constant(D_resid)
    ols_fit = sm.OLS(Y_resid, X_ols).fit(cov_type="HC3")

    beta1_hat = ols_fit.params[1]
    se = ols_fit.bse[1]

    z = norm.ppf(0.975)  # 95%
    ci_low, ci_high = beta1_hat - z * se, beta1_hat + z * se

    return {
        "beta1_hat": float(beta1_hat),
        "se_HC3": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "k_y": int(np.sum(model_y.coef_ != 0)),
        "k_d": int(np.sum(model_d.coef_ != 0)),
        "alpha": float(alpha),
    }
