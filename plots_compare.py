from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(folder: Path) -> pd.DataFrame:
    path = folder / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found in {folder}")
    return pd.read_csv(path)


def plot_metric(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    scenarios = df["scenario"]
    width = 0.35
    x = range(len(scenarios))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df[f"{metric}_lasso"], width, label="Double LASSO")
    ax.bar([i + width / 2 for i in x], df[f"{metric}_ols"], width, label="OLS")

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    outpath = outdir / f"{metric}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare results between Double LASSO and OLS summaries.")
    parser.add_argument("--lasso_dir", type=Path, default=Path("results"))
    parser.add_argument("--ols_dir", type=Path, default=Path("results_ols"))
    parser.add_argument("--outdir", type=Path, default=Path("cplots"))
    args = parser.parse_args()

    lasso_df = load_summary(args.lasso_dir)
    ols_df = load_summary(args.ols_dir)

    merged = pd.merge(
        lasso_df,
        ols_df,
        on="scenario",
        suffixes=("_lasso", "_ols"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping scenarios between the two summary files.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "coverage",
        "bias",
        "rmse",
        "ci_length_mean",
        "treatment_effect_hat_mean",
        "standard_error_HC3_mean",
    ]
    for metric in metrics:
        if f"{metric}_lasso" in merged and f"{metric}_ols" in merged:
            plot_metric(merged[["scenario", f"{metric}_lasso", f"{metric}_ols"]], metric, args.outdir)

    print(f"Saved comparison plots to {args.outdir}")


if __name__ == "__main__":
    main()
