
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

from protocols import DGPProtocol, EstimatorProtocol
from estimators.lasso import plugin_alpha

def run_simulation(*,
                   R: int = 500,
                   n: int = 200,
                   p: int = 100,
                   s: int = 5,
                   beta1: float = 2.0,
                   rho: float = 0.2,
                   ci_level: float = 0.95,
                   c: float = 1.1,
                   seed: int = 123,
                   dgp: DGPProtocol,
                   estimator: EstimatorProtocol) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(R):
        Y, D, X = dgp(n=n, p=p, s=s, beta1=beta1, rho=rho, seed=rng.integers(0, 1_000_000))
        alpha = plugin_alpha(n, p, c=c)
        est = estimator(Y, D, X, alpha=alpha, ci_level=ci_level)
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
            "alpha": est["alpha"],
        })
    return pd.DataFrame(rows)
