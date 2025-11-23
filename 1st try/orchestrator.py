from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runner import run_simulation
from protocols import DGPProtocol, EstimatorProtocol

def sweep_designs(*,
                  param: str = "n",
                  values=(100, 200, 400),
                  base: dict = dict(n=200, p=100, s=5, beta1=2.0, rho=0.2),
                  R: int = 500,
                  ci_level: float = 0.95,
                  c: float = 1.1,
                  seed: int = 123,
                  use_cv: bool = False,
                  dgp: DGPProtocol = None,
                  estimator: EstimatorProtocol = None,
                  make_plot: bool = True,
                  save_csv: str | None = None) -> pd.DataFrame:
    """
    Coverage vs CI-length sweep over one design parameter.
    Same calculation as in the single-file script.
    """
    rows = []
    for val in values:
        kwargs = dict(base)
        kwargs[param] = val
        df = run_simulation(R=R, ci_level=ci_level, c=c, seed=seed, use_cv=use_cv, dgp=dgp, estimator=estimator, **kwargs)
        beta1 = kwargs.get("beta1", 2.0)
        coverage = df["covered"].mean()
        avg_len = df["ci_length"].mean()
        bias = (df["beta1_hat"] - beta1).mean()
        rmse = np.sqrt(((df["beta1_hat"] - beta1)**2).mean())
        rows.append({
            "param": param,
            "value": val,
            "coverage": coverage,
            "avg_ci_length": avg_len,
            "bias": bias,
            "rmse": rmse,
            "avg_k_y": df["k_y"].mean(),
            "avg_k_d": df["k_d"].mean(),
        })
    out = pd.DataFrame(rows)
    if save_csv:
        out.to_csv(save_csv, index=False)

    if make_plot:
        plt.figure()
        plt.plot(out["avg_ci_length"], out["coverage"], marker="o")
        for _, row in out.iterrows():
            plt.annotate(str(row["value"]), (row["avg_ci_length"], row["coverage"]),
                         textcoords="offset points", xytext=(6, 6), fontsize=9)
        plt.xlabel("Average CI length")
        plt.ylabel("Coverage")
        plt.title(f"Coverage vs CI length (sweep {param})")
        plt.grid(True, alpha=0.3)
        plt.show()
    return out
