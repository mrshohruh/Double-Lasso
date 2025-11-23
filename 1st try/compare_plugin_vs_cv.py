import glob
import os

import numpy as np
import pandas as pd


TRUE_BETA1 = 2.0
REQUIRED_COLS = {"covered", "ci_length", "beta1_hat", "k_y", "k_d"}


def summarize_dir(results_dir: str) -> pd.DataFrame:
    """
    Summarize all raw scenario CSVs in a given directory.
    Expects files with columns: covered, ci_length, beta1_hat, k_y, k_d.
    Ignores any CSVs that don't have these columns (e.g. summary files).
    """
    pattern = os.path.join(results_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise SystemExit(f"No CSV files found in {results_dir}/")

    rows = []
    for path in files:
        # Peek at the header to see if it's a raw simulation file
        df_head = pd.read_csv(path, nrows=1)
        if not REQUIRED_COLS.issubset(df_head.columns):
            continue

        df = pd.read_csv(path)
        name = os.path.splitext(os.path.basename(path))[0]

        coverage = df["covered"].mean()
        avg_ci_length = df["ci_length"].mean()
        bias = (df["beta1_hat"] - TRUE_BETA1).mean()
        rmse = np.sqrt(((df["beta1_hat"] - TRUE_BETA1) ** 2).mean())
        avg_k_y = df["k_y"].mean()
        avg_k_d = df["k_d"].mean()

        rows.append(
            dict(
                scenario=name,
                coverage=coverage,
                avg_ci_length=avg_ci_length,
                bias=bias,
                rmse=rmse,
                avg_k_y=avg_k_y,
                avg_k_d=avg_k_d,
            )
        )

    if not rows:
        raise SystemExit(
            f"No valid scenario files with required columns {REQUIRED_COLS} found in {results_dir}/"
        )

    return pd.DataFrame(rows).sort_values("scenario")


def main():
    # Ensure both result directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results_cv", exist_ok=True)

    # Summaries for plugin-penalty runs and CV-penalty runs
    plugin = summarize_dir("results")
    cv = summarize_dir("results_cv")

    # Merge on scenario name
    merged = plugin.merge(cv, on="scenario", suffixes=("_plugin", "_cv"))

    # Add difference columns: CV - plugin
    merged["coverage_diff"] = merged["coverage_cv"] - merged["coverage_plugin"]
    merged["bias_diff"] = merged["bias_cv"] - merged["bias_plugin"]
    merged["avg_ci_length_diff"] = (
        merged["avg_ci_length_cv"] - merged["avg_ci_length_plugin"]
    )

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "plugin_vs_cv_summary.csv")
    merged.to_csv(out_path, index=False)

    print("Wrote plugin vs CV comparison to:", out_path)
    print(merged)


if __name__ == "__main__":
    main()
