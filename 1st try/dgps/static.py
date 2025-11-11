
import numpy as np

def simulate_dgp(n=200, p=100, s=5, beta1=2.0, rho=0.0, seed=None):
    """
    DGP:
        X ~ N(0, Sigma) where Sigma has off-diagonal rho
        D = sum_{j<=s} X_j + v
        Y = beta1 * D + sum_{j<=s} X_j + u
    """
    rng = np.random.default_rng(seed)
    if rho == 0.0:
        X = rng.normal(size=(n, p))
    else:
        # Equicorrelated Sigma = (1-rho) I + rho * 11'
        L = np.linalg.cholesky((1 - rho) * np.eye(p) + rho * np.ones((p, p)))
        X = rng.normal(size=(n, p)) @ L.T

    signal_x = X[:, :s].sum(axis=1)
    v = rng.normal(size=n)
    u = rng.normal(size=n)

    D = signal_x + v
    Y = beta1 * D + signal_x + u
    return Y, D, X
