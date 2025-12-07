import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.api as sm
from scipy.stats import norm


def _plugin_alpha_from_sigma(n, p, sigma_hat, c=1.1, alpha_level=0.1):
    """
    Compute sklearn's alpha from a given sigma_hat using the plug-in rule
    from Chernozhukov et al. (Chapter 3).

    Book-style Lasso problem:
        sum_i (Y_i - X_i' b)^2 + lambda * sum_j |b_j|

    Plug-in rule (upper bound form):
        lambda ~ 2 * c * sigma_hat * sqrt(2 * n * log(2p / alpha_level))

    sklearn solves:
        (1/(2n)) ||Y - Xb||^2_2 + alpha * ||b||_1

    Matching coefficients:
        lambda = 2n * alpha  =>  alpha = lambda / (2n).

    So:
        alpha_sklearn = c * sigma_hat * sqrt(2 * log(2p / alpha_level) / n)
    """
    if p <= 0:
        raise ValueError("Number of predictors p must be positive.")
    if not (0 < alpha_level < 1):
        raise ValueError("alpha_level must be in (0, 1).")

    # alpha for sklearn's Lasso
    alpha_sklearn = (
        c * sigma_hat * np.sqrt(2.0 * np.log(2.0 * p / alpha_level) / n)
    )
    return alpha_sklearn


def plugin_alpha(X, y, c=1.1, alpha_level=0.1, max_iter=1, tol=1e-4):
    """
    Plug-in style L1 penalty for sklearn's Lasso:
        (1/(2n)) ||y - Xb||^2 + alpha ||b||_1

    This implements the plug-in rule from Chernozhukov et al. (Ch. 3),
    with a simple iterative update of sigma_hat as in Section 3.A.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response vector.
    c : float
        Multiplicative constant in the penalty rule (typically slightly > 1).
    alpha_level : float
        Tail probability parameter 'a' in log(2p/a). Default ~ 0.1.
    max_iter : int
        Maximum number of iterations for refining sigma_hat. In practice,
        1-2 iterations are usually enough (book suggests K = 1 works well).
    tol : float
        Tolerance for convergence of sigma_hat.

    Returns
    -------
    alpha : float
        Penalty parameter in sklearn scale to be passed to Lasso(alpha=...).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, p = X.shape

    # Initial sigma_hat: residual std from intercept-only model
    y_centered = y - y.mean()
    sigma_hat = float(np.sqrt(np.mean(y_centered ** 2)))

    if sigma_hat <= 0:
        # Degenerate case; fall back to 1.0 to avoid crashes
        sigma_hat = 1.0

    for _ in range(max_iter):
        alpha_current = _plugin_alpha_from_sigma(
            n=n,
            p=p,
            sigma_hat=sigma_hat,
            c=c,
            alpha_level=alpha_level,
        )

        # Fit Lasso with current alpha to update sigma_hat
        model = Lasso(alpha=alpha_current, fit_intercept=True, max_iter=5000)
        model.fit(X, y)
        resid = y - model.predict(X)
        sigma_new = float(np.sqrt(np.mean(resid ** 2)))

        if abs(sigma_new - sigma_hat) <= tol:
            sigma_hat = sigma_new
            break
        sigma_hat = sigma_new

    # Final alpha based on final sigma_hat
    alpha_final = _plugin_alpha_from_sigma(
        n=n,
        p=p,
        sigma_hat=sigma_hat,
        c=c,
        alpha_level=alpha_level,
    )
    return alpha_final


def cv_alpha(X, y):
    """
    Compute alpha via 10-fold cross-validation (sklearn's LassoCV).
    Returns the selected alpha_.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    model = LassoCV(cv=10, fit_intercept=True, n_jobs=-1)
    model.fit(X, y)
    return float(model.alpha_)


def lasso_residuals(X, y, alpha):
    """
    Fit Lasso(y ~ X) and return residuals y - (intercept + X b_hat),
    along with the fitted model.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    model.fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat
    return resid, model


def double_lasso_ci(Y, D, X, alpha=None, ci_level=0.95, use_cv=False,
                    plugin_c=1.1, plugin_alpha_level=0.1):
    """
    Double LASSO confidence interval for the coefficient of D in:
        Y = beta1 * D + X * gamma + eps

    Steps:
      1) Lasso Y ~ X -> residuals Y~.
      2) Lasso D ~ X -> residuals D~.
      3) OLS Y~ on D~, HC3 robust SE -> CI for beta1.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Treatment / target regressor.
    X : array-like, shape (n, p)
        Controls.
    alpha : float or None
        If not None and use_cv=False, this value is used as the Lasso alpha
        for both Y and D equations (sklearn scale).
    ci_level : float
        Confidence level (e.g., 0.95).
    use_cv : bool
        If True, use LassoCV to select alpha for Y and D separately.
        If False and alpha is None, use the plug-in rule (Chernozhukov et al.).
    plugin_c : float
        Constant c used in the plug-in penalty rule (typically ~1.1).
    plugin_alpha_level : float
        Tail probability 'a' in log(2p/a) in the plug-in rule.

    Returns
    -------
    results : dict with keys
        - beta1_hat : float
        - se_HC3    : float
        - ci_low    : float
        - ci_high   : float
        - k_y       : int   (number of selected controls in Y~ regression)
        - k_d       : int   (number of selected controls in D~ regression)
        - alpha_y   : float (alpha used for Y regression)
        - alpha_d   : float (alpha used for D regression)
    """
    Y = np.asarray(Y)
    D = np.asarray(D)
    X = np.asarray(X)
    n, p = X.shape

    # Choose penalty levels
    if use_cv:
        alpha_y = cv_alpha(X, Y)
        alpha_d = cv_alpha(X, D)
    else:
        if alpha is not None:
            # User-specified alpha (same for both equations)
            alpha_y = alpha_d = float(alpha)
        else:
            # True plug-in rule, potentially different for Y and D equations
            alpha_y = plugin_alpha(
                X, Y, c=plugin_c, alpha_level=plugin_alpha_level
            )
            alpha_d = plugin_alpha(
                X, D, c=plugin_c, alpha_level=plugin_alpha_level
            )

    # Step 1: partial out X from Y
    Y_resid, model_y = lasso_residuals(X, Y, alpha_y)

    # Step 2: partial out X from D
    D_resid, model_d = lasso_residuals(X, D, alpha_d)

    # Step 3: OLS on residuals with HC3
    X_ols = sm.add_constant(D_resid)
    ols_fit = sm.OLS(Y_resid, X_ols).fit(cov_type="HC3")

    beta1_hat = float(ols_fit.params[1])
    se = float(ols_fit.bse[1])

    # z-quantile for desired CI level
    z = norm.ppf(0.5 + ci_level / 2.0)
    ci_low = beta1_hat - z * se
    ci_high = beta1_hat + z * se

    return {
        "beta1_hat": beta1_hat,
        "se_HC3": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "k_y": int(np.sum(model_y.coef_ != 0)),
        "k_d": int(np.sum(model_d.coef_ != 0)),
        "alpha_y": float(alpha_y),
        "alpha_d": float(alpha_d),
    }
