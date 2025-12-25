from __future__ import annotations
import numpy as np
import pandas as pd

from runner import run_simulation
from protocols import DGPProtocol, EstimatorProtocol

def sweep_designs(*,
                  param: str = "n_samples",
                  values=(120, 200, 320),
                  base: dict = dict(
                      n_samples=200,
                      n_covariates=400,
                      n_relevant_covariates=5,
                      treatment_effect=2.0,
                      covariate_correlation=0.2,
                  ),
                  R: int = 500,
                  ci_level: float = 0.95,
                  plugin_c: float = 0.6,
                  seed: int = 123,
                  use_cv: bool = False,
                  dgp: DGPProtocol = None,
                  estimator: EstimatorProtocol = None,
                  save_csv: str | None = None) -> pd.DataFrame:
    """
    Coverage vs CI-length sweep over one design parameter.
    Same calculation as in the single-file script.
    """
    rows = []
    for val in values:
        kwargs = dict(base)
        kwargs[param] = val
        df = run_simulation(
            R=R,
            ci_level=ci_level,
            plugin_c=plugin_c,
            seed=seed,
            use_cv=use_cv,
            dgp=dgp,
            estimator=estimator,
            **kwargs,
        )
        treatment_effect = kwargs.get("treatment_effect", 2.0)
        coverage = df["covered"].mean()
        avg_len = df["ci_length"].mean()
        bias = (df["treatment_effect_hat"] - treatment_effect).mean()
        rmse = np.sqrt(((df["treatment_effect_hat"] - treatment_effect) ** 2).mean())
        rows.append({
            "param": param,
            "value": val,
            "coverage": coverage,
            "avg_ci_length": avg_len,
            "bias": bias,
            "rmse": rmse,
            "avg_n_selected_outcome_controls": df["n_selected_outcome_controls"].mean(),
            "avg_n_selected_treatment_controls": df["n_selected_treatment_controls"].mean(),
        })
    out = pd.DataFrame(rows)
    if save_csv:
        out.to_csv(save_csv, index=False)
    return out
