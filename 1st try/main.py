from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
import numpy as np

from dgps.static import simulate_dgp as static_dgp
from dgps.static_easier import simulate_dgp as static_easier_dgp
from dgps.heavy_tail_dgp import simulate_dgp as heavy_tail_dgp
from estimators.lasso import double_lasso_ci
from estimators.ols import ols_ci
from runner import run_simulation
from orchestrator import sweep_designs
from scenarios import get_scenarios

DGP_MAP = {
    "static": static_dgp,
    "static_easier": static_easier_dgp,
    "heavy_tail": heavy_tail_dgp,
}

ESTIMATOR_MAP = {
    "double_lasso": double_lasso_ci,
    "ols": ols_ci,
}


def resolve_dgp(name: str):
    try:
        return DGP_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dgp '{name}'. Options: {', '.join(DGP_MAP)}") from exc


def resolve_estimator(name: str):
    try:
        return ESTIMATOR_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown estimator '{name}'. Options: {', '.join(ESTIMATOR_MAP)}") from exc


PARAM_NAME_MAP = {
    "n": "n_samples",
    "p": "n_covariates",
    "s": "n_relevant_covariates",
    "rho": "covariate_correlation",
    "n_samples": "n_samples",
    "n_covariates": "n_covariates",
    "n_relevant_covariates": "n_relevant_covariates",
    "covariate_correlation": "covariate_correlation",
}


def write_summary(folder: str, beta1_true: float = 2.0, scenario_names: list[str] | None = None) -> pd.DataFrame:
    """Aggregate per-scenario CSVs in a folder into summary.csv.

    If scenario_names is provided, only include CSVs whose stem is in that list.
    """
    rows = []
    def pick_column(df: pd.DataFrame, new: str, old: str, fill_value=np.nan) -> pd.Series:
        if new in df:
            return df[new]
        if old in df:
            return df[old]
        return pd.Series([fill_value] * len(df))

    for path in glob.glob(os.path.join(folder, "*.csv")):
        if os.path.basename(path) == "summary.csv":
            continue
        df = pd.read_csv(path)
        scenario = os.path.splitext(os.path.basename(path))[0]
        if scenario_names is not None and scenario not in scenario_names:
            continue
        treatment_hat_series = pick_column(df, "treatment_effect_hat", "beta1_hat")
        bias_series = treatment_hat_series - beta1_true
        rows.append({
            "scenario": scenario,
            "rows": len(df),
            "coverage": df["covered"].mean(),
            "treatment_effect_hat_mean": treatment_hat_series.mean(),
            "bias": bias_series.mean(),
            "rmse": np.sqrt((bias_series ** 2).mean()),
            "ci_length_mean": df["ci_length"].mean(),
            "n_selected_outcome_controls_mean": pick_column(df, "n_selected_outcome_controls", "k_y").mean(),
            "n_selected_treatment_controls_mean": pick_column(df, "n_selected_treatment_controls", "k_d").mean(),
            "outcome_lasso_penalty_mean": pick_column(df, "outcome_lasso_penalty", "alpha_y").mean(),
            "treatment_lasso_penalty_mean": pick_column(df, "treatment_lasso_penalty", "alpha_d").mean(),
            "standard_error_HC3_mean": pick_column(df, "standard_error_HC3", "se_HC3").mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(folder, "summary.csv"), index=False)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Double LASSO simulations (modular layout).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # simple run
    runp = sub.add_parser("run", help="Run a single design R times.")
    runp.add_argument("--R", type=int, default=500, help="Number of Monte Carlo replications")
    runp.add_argument("--n", type=int, default=200)
    runp.add_argument("--p", type=int, default=240)
    runp.add_argument("--s", type=int, default=5)
    runp.add_argument("--beta1", type=float, default=2.0)
    runp.add_argument("--rho", type=float, default=0.2)
    runp.add_argument("--c", type=float, default=0.6)
    runp.add_argument("--ci", type=float, default=0.95)
    runp.add_argument("--seed", type=int, default=123)
    runp.add_argument("--out", type=str, default="results.csv")
    runp.add_argument("--dgp", type=str, default="static", choices=list(DGP_MAP))
    runp.add_argument("--estimator", type=str, default="double_lasso", choices=list(ESTIMATOR_MAP))
    runp.add_argument("--use_cv", action="store_true",
                      help="Use cross-validated alpha instead of plugin alpha.")

    # sweep
    swp = sub.add_parser("sweep", help="Sweep over one design parameter.")
    swp.add_argument("--param", type=str, choices=list(PARAM_NAME_MAP), default="n")
    swp.add_argument("--values", type=str, default="120,200,320")
    swp.add_argument("--R", type=int, default=500, help="Number of Monte Carlo replications")
    swp.add_argument("--n", type=int, default=200)
    swp.add_argument("--p", type=int, default=400)
    swp.add_argument("--s", type=int, default=5)
    swp.add_argument("--beta1", type=float, default=2.0)
    swp.add_argument("--rho", type=float, default=0.2)
    swp.add_argument("--c", type=float, default=0.6)
    swp.add_argument("--ci", type=float, default=0.95)
    swp.add_argument("--seed", type=int, default=123)
    swp.add_argument("--out", type=str, default="sweep_results.csv")
    swp.add_argument("--dgp", type=str, default="static", choices=list(DGP_MAP))
    swp.add_argument("--estimator", type=str, default="double_lasso", choices=list(ESTIMATOR_MAP))
    swp.add_argument("--use_cv", action="store_true",
                     help="Use cross-validated alpha instead of plugin alpha.")

    # scenarios
    scp = sub.add_parser("scenarios", help="Run predefined scenarios.")
    scp.add_argument("--outdir", type=str, default="results")
    scp.add_argument("--dgp", type=str, default="static", choices=list(DGP_MAP))
    scp.add_argument("--estimator", type=str, default="double_lasso", choices=list(ESTIMATOR_MAP))
    scp.add_argument("--scenarios", type=str, default="",
                     help="Comma-separated scenario names to run; defaults to all.")
    scp.add_argument("--use_cv", action="store_true",
                     help="Use cross-validated alpha instead of plugin alpha.")

    args = parser.parse_args()
    dgp = resolve_dgp(getattr(args, "dgp", "static"))
    estimator = resolve_estimator(getattr(args, "estimator", "double_lasso"))

    if args.cmd == "run":
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        df = run_simulation(
            R=args.R,
            n_samples=args.n,
            n_covariates=args.p,
            n_relevant_covariates=args.s,
            treatment_effect=args.beta1,
            covariate_correlation=args.rho,
            ci_level=args.ci,
            plugin_c=args.c,
            seed=args.seed,
            use_cv=args.use_cv,
            dgp=dgp,
            estimator=estimator,
        )
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

    elif args.cmd == "sweep":
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        canonical_param = PARAM_NAME_MAP[args.param]
        base = dict(
            n_samples=args.n,
            n_covariates=args.p,
            n_relevant_covariates=args.s,
            treatment_effect=args.beta1,
            covariate_correlation=args.rho,
        )
        value_type = type(base[canonical_param])
        values = [value_type(v) for v in args.values.split(",")]
        sweep = sweep_designs(
            param=canonical_param,
            values=values,
            R=args.R,
            ci_level=args.ci,
            plugin_c=args.c,
            seed=args.seed,
            use_cv=args.use_cv,
            dgp=dgp,
            estimator=estimator,
            save_csv=args.out,
        )
        print(sweep)
        print(f"Saved sweep to {args.out}")

    elif args.cmd == "scenarios":
        import os
        outdir = args.outdir
        if args.dgp == "heavy_tail" and args.outdir == "results":
            # Keep heavy-tailed runs separate from the Gaussian baseline outputs
            outdir = "results_heavy"
            print("heavy_tail DGP detected: writing scenarios to results_heavy/")
        os.makedirs(outdir, exist_ok=True)
        scenarios = get_scenarios()
        selected = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if selected:
            scenarios = [sc for sc in scenarios if sc.name in selected]
            if not scenarios:
                raise ValueError(f"No matching scenarios for names: {selected}")
        for sc in scenarios:
            scenario_kwargs = dict(
                R=sc.R,
                n_samples=sc.n_samples,
                n_covariates=sc.n_covariates,
                n_relevant_covariates=sc.n_relevant_covariates,
                treatment_effect=sc.treatment_effect,
                covariate_correlation=sc.covariate_correlation,
                ci_level=sc.ci_level,
                plugin_c=sc.plugin_c,
                seed=sc.seed,
                use_cv=args.use_cv,
            )
            df = run_simulation(
                dgp=dgp,
                estimator=estimator,
                **scenario_kwargs,
            )
            out = f"{outdir}/{sc.name}.csv"
            df.to_csv(out, index=False)
            print(f"Saved {sc.name} -> {out}")
        scenario_names = [sc.name for sc in scenarios]
        summary = write_summary(outdir, beta1_true=scenarios[0].treatment_effect, scenario_names=scenario_names)
        print("Updated summary.csv:")
        print(summary)

if __name__ == "__main__":
    main()
