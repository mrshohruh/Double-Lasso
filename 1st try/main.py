
import argparse
import pandas as pd

from dgps.static import simulate_dgp
from estimators.lasso import double_lasso_ci
from runner import run_simulation
from orchestrator import sweep_designs
from scenarios import get_scenarios

def main():
    parser = argparse.ArgumentParser(description="Double LASSO simulations (modular layout).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # simple run
    runp = sub.add_parser("run", help="Run a single design R times.")
    runp.add_argument("--R", type=int, default=100)
    runp.add_argument("--n", type=int, default=200)
    runp.add_argument("--p", type=int, default=100)
    runp.add_argument("--s", type=int, default=5)
    runp.add_argument("--beta1", type=float, default=2.0)
    runp.add_argument("--rho", type=float, default=0.2)
    runp.add_argument("--c", type=float, default=1.1)
    runp.add_argument("--ci", type=float, default=0.95)
    runp.add_argument("--seed", type=int, default=123)
    runp.add_argument("--out", type=str, default="results.csv")

    # sweep
    swp = sub.add_parser("sweep", help="Sweep over one design parameter.")
    swp.add_argument("--param", type=str, choices=["n", "p", "s", "rho"], default="n")
    swp.add_argument("--values", type=str, default="80,120,200,320")
    swp.add_argument("--R", type=int, default=100)
    swp.add_argument("--n", type=int, default=200)
    swp.add_argument("--p", type=int, default=100)
    swp.add_argument("--s", type=int, default=5)
    swp.add_argument("--beta1", type=float, default=2.0)
    swp.add_argument("--rho", type=float, default=0.2)
    swp.add_argument("--c", type=float, default=1.1)
    swp.add_argument("--ci", type=float, default=0.95)
    swp.add_argument("--seed", type=int, default=123)
    swp.add_argument("--out", type=str, default="sweep_results.csv")

    # scenarios
    scp = sub.add_parser("scenarios", help="Run predefined scenarios.")
    scp.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    if args.cmd == "run":
        df = run_simulation(
            R=args.R, n=args.n, p=args.p, s=args.s, beta1=args.beta1, rho=args.rho,
            ci_level=args.ci, c=args.c, seed=args.seed,
            dgp=simulate_dgp, estimator=double_lasso_ci
        )
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

    elif args.cmd == "sweep":
        values = [type(getattr(args, args.param))(v) for v in args.values.split(",")]
        base = dict(n=args.n, p=args.p, s=args.s, beta1=args.beta1, rho=args.rho)
        sweep = sweep_designs(
            param=args.param, values=values, R=args.R,
            ci_level=args.ci, c=args.c, seed=args.seed,
            dgp=simulate_dgp, estimator=double_lasso_ci,
            make_plot=True, save_csv=args.out
        )
        print(sweep)
        print(f"Saved sweep to {args.out}")

    elif args.cmd == "scenarios":
        import os
        os.makedirs(args.outdir, exist_ok=True)
        for sc in get_scenarios():
            df = run_simulation(
                R=sc.R, n=sc.n, p=sc.p, s=sc.s, beta1=sc.beta1, rho=sc.rho,
                ci_level=sc.ci_level, c=sc.c, seed=sc.seed,
                dgp=simulate_dgp, estimator=double_lasso_ci
            )
            out = f"{args.outdir}/{sc.name}.csv"
            df.to_csv(out, index=False)
            print(f"Saved {sc.name} -> {out}")

if __name__ == "__main__":
    main()
