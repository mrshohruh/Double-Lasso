import glob
import os

import numpy as np
import pandas as pd


TRUE_BETA1 = 2.0
RESULTS_DIR = "results"


def summarize_scenario(path: str) -> dict:
    """
    Read one scenario CSV and compute summary stats.
    """
    df = pd.read_csv(path)

    name = os.path.splitext(os.path.basename(path))[0]

    coverage = df["covered"].mean()
    avg_ci_length = df["ci_length"].mean()
    bias = (df["beta1_hat"] - TRUE_BETA1).mean()
    rmse = np.sqrt(((df["beta1_hat"] - TRUE_BETA1) ** 2).mean())
    avg_k_y = df["k_y"].mean()
    avg_k_d = df["k_d"].mean()

    return dict(
        scenario=name,
        coverage=coverage,
        avg_ci_length=avg_ci_length,
        bias=bias,
        rmse=rmse,
        avg_k_y=avg_k_y,
        avg_k_d=avg_k_d,
    )


def main():
    # Read all scenario result CSVs in the results/ folder
    pattern = os.path.join(RESULTS_DIR, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise SystemExit(f"No CSV files found in {RESULTS_DIR}/")

    rows = [summarize_scenario(path) for path in files]
    summary = pd.DataFrame(rows)

    # Sort by scenario name for readability
    summary = summary.sort_values("scenario")

    out_path = os.path.join(RESULTS_DIR, "scenario_summary.csv")
    summary.to_csv(out_path, index=False)

    print("Wrote summary to:", out_path)
    print(summary)


if __name__ == "__main__":
    main()
