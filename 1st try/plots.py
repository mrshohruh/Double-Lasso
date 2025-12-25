from __future__ import annotations

"""
Coverage probability plots for Double LASSO simulation summaries.

Expected inputs: summary.csv under
  - results
  - results_cv
  - results_easierdgp
  - results_ecv (optional)
  - results_heavy (optional)

Outputs are written to plots/:
  - coverage_by_scenario.png
  - coverage_vs_rho.png
  - coverage_vs_n.png
  - coverage_vs_p.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PLOTS_DIR = Path("plots")

RESULT_FOLDERS = {
    "results": Path("results"),
    "results_cv": Path("results_cv"),
    "results_easierdgp": Path("results_easierdgp"),
    "results_ecv": Path("results_ecv"),
    "results_heavy": Path("results_heavy"),
    "results_heavycv": Path("results_heavycv"),
}

LABELS = {
    "results": "Plugin alpha (static dgp)",
    "results_cv": "Cross-validated alpha (static dgp)",
    "results_easierdgp": "Plugin alpha (easier dgp)",
    "results_ecv": "Cross-validated alpha (easier dgp)",
    "results_heavy": "Plugin alpha (heavy-tailed dgp)",
    "results_heavycv": "Cross-validated alpha (heavy-tailed dgp)",
}

# Preserve a consistent legend order across all plots
LABEL_ORDER = [
    LABELS["results"],
    LABELS["results_cv"],
    LABELS["results_easierdgp"],
    LABELS["results_ecv"],
    LABELS["results_heavy"],
    LABELS["results_heavycv"],
]

COLORS = {
    "Plugin alpha (static dgp)": "#1f77b4",
    "Cross-validated alpha (static dgp)": "#d62728",
    "Plugin alpha (easier dgp)": "#2ca02c",
    "Cross-validated alpha (easier dgp)": "#9467bd",
    "Plugin alpha (heavy-tailed dgp)": "#ff7f0e",
    "Cross-validated alpha (heavy-tailed dgp)": "#8c564b",
}

# Scenario metadata based on scenarios.py
SCENARIO_PARAMS = {
    "small_corr_0_2": dict(n_samples=120, n_covariates=150, covariate_correlation=0.2),
    "small_corr_0_0": dict(n_samples=120, n_covariates=150, covariate_correlation=0.0),
    "small_corr_0_5": dict(n_samples=120, n_covariates=150, covariate_correlation=0.5),
    "medium_corr_0_2": dict(n_samples=200, n_covariates=240, covariate_correlation=0.2),
    "medium_corr_0_0": dict(n_samples=200, n_covariates=240, covariate_correlation=0.0),
    "medium_corr_0_5": dict(n_samples=200, n_covariates=240, covariate_correlation=0.5),
    "large_corr_0_2": dict(n_samples=320, n_covariates=384, covariate_correlation=0.2),
    "large_corr_0_0": dict(n_samples=320, n_covariates=384, covariate_correlation=0.0),
    "large_corr_0_5": dict(n_samples=320, n_covariates=384, covariate_correlation=0.5),
    # Legacy scenario names for backward compatibility with older CSVs
    "small": dict(n_samples=120, n_covariates=150, covariate_correlation=0.2),
    "small_rho_0": dict(n_samples=120, n_covariates=150, covariate_correlation=0.0),
    "small_rho_0_5": dict(n_samples=120, n_covariates=150, covariate_correlation=0.5),
    "medium": dict(n_samples=200, n_covariates=240, covariate_correlation=0.2),
    "medium_rho_0": dict(n_samples=200, n_covariates=240, covariate_correlation=0.0),
    "medium_rho_0_5": dict(n_samples=200, n_covariates=240, covariate_correlation=0.5),
    "large": dict(n_samples=320, n_covariates=384, covariate_correlation=0.2),
    "large_rho_0": dict(n_samples=320, n_covariates=384, covariate_correlation=0.0),
    "large_rho_0_5": dict(n_samples=320, n_covariates=384, covariate_correlation=0.5),
}

SCENARIO_ORDER = [
    "large_corr_0_0",
    "large_corr_0_2",
    "large_corr_0_5",
    "medium_corr_0_0",
    "medium_corr_0_2",
    "medium_corr_0_5",
    "small_corr_0_0",
    "small_corr_0_2",
    "small_corr_0_5",
    "classical_low_dim",
    "near_p_equals_n",
    "p_equals_n",
    # Legacy aliases
    "large",
    "large_rho_0",
    "large_rho_0_5",
    "medium",
    "medium_rho_0",
    "medium_rho_0_5",
    "small",
    "small_rho_0",
    "small_rho_0_5",
]


def load_run_data() -> pd.DataFrame:
    """
    Load per-replication CSVs (excluding summary.csv) for CI length distributions.
    """
    rows = []
    column_aliases = {
        "treatment_effect_hat": ["treatment_effect_hat", "beta1_hat"],
        "ci_length": ["ci_length"],
        "n_selected_outcome_controls": ["n_selected_outcome_controls", "k_y"],
        "n_selected_treatment_controls": ["n_selected_treatment_controls", "k_d"],
    }
    for key, folder in RESULT_FOLDERS.items():
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            if path.name == "summary.csv":
                continue
            df = pd.read_csv(path)
            scenario = path.stem
            df = df.assign(source=key, label=LABELS.get(key, key), scenario=scenario)
            selected_cols: list[str] = []
            rename_map: dict[str, str] = {}
            for canonical_name, aliases in column_aliases.items():
                for alias in aliases:
                    if alias in df.columns:
                        selected_cols.append(alias)
                        rename_map[alias] = canonical_name
                        break
            if selected_cols:
                standardized = df[selected_cols + ["label", "scenario"]].rename(columns=rename_map)
                rows.append(standardized)
    if not rows:
        return pd.DataFrame(columns=[
            "treatment_effect_hat",
            "ci_length",
            "n_selected_outcome_controls",
            "n_selected_treatment_controls",
            "label",
            "scenario",
        ])
    return pd.concat(rows, ignore_index=True)


def _ordered_labels(labels: list[str]) -> list[str]:
    """Return labels in the global LABEL_ORDER, dropping missing ones."""
    present = set(labels)
    return [lbl for lbl in LABEL_ORDER if lbl in present]


def _ordered_scenarios(scenarios: list[str]) -> list[str]:
    """Return scenarios in the global SCENARIO_ORDER, dropping missing ones."""
    present = set(scenarios)
    return [sc for sc in SCENARIO_ORDER if sc in present]


def load_summaries() -> pd.DataFrame:
    frames = []
    for key, folder in RESULT_FOLDERS.items():
        summary_path = folder / "summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        rename_map = {
            "beta1_hat_mean": "treatment_effect_hat_mean",
            "k_y_mean": "n_selected_outcome_controls_mean",
            "k_d_mean": "n_selected_treatment_controls_mean",
            "alpha_y_mean": "outcome_lasso_penalty_mean",
            "alpha_d_mean": "treatment_lasso_penalty_mean",
            "se_HC3_mean": "standard_error_HC3_mean",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["source"] = key
        df["label"] = LABELS.get(key, key)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No summary.csv files found in configured result folders.")
    out = pd.concat(frames, ignore_index=True)
    # Backfill missing scenario metadata with NaNs to avoid shape errors
    scenario_meta = out["scenario"].map(SCENARIO_PARAMS).apply(
        lambda meta: meta if isinstance(meta, dict) else {"n_samples": np.nan, "n_covariates": np.nan, "covariate_correlation": np.nan}
    )
    out[["n_samples", "n_covariates", "covariate_correlation"]] = scenario_meta.apply(pd.Series)
    return out


def plot_coverage_by_scenario(df: pd.DataFrame, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values="coverage")
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage across scenarios")
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    ax.legend(title="Configuration", frameon=False)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def plot_metric_by_scenario(df: pd.DataFrame, metric: str, ylabel: str, title: str, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values=metric)
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    ax.legend(title="Configuration", frameon=False)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def line_plot(df: pd.DataFrame, x: str, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in _ordered_labels(df["label"].unique().tolist()):
        group = df[df["label"] == label]
        sorted_group = group.sort_values(x)
        ax.plot(sorted_group[x], sorted_group["coverage"], marker="o",
                label=label, color=COLORS.get(label, None))
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(x)
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_coverage_vs_rho(df: pd.DataFrame, outfile: Path) -> None:
    # Use all available rho values; average over n/p within each label-rho
    agg = df.groupby(["label", "covariate_correlation"], as_index=False)["coverage"].mean()
    line_plot(agg, x="covariate_correlation", title="Coverage vs correlation (rho)", outfile=outfile)


def plot_coverage_vs_n(df: pd.DataFrame, outfile: Path) -> None:
    # Focus on rho=0.2 to align across sizes
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_samples"], as_index=False)["coverage"].mean()
    line_plot(agg, x="n_samples", title="Coverage vs sample size (rho=0.2)", outfile=outfile)


def plot_coverage_vs_p(df: pd.DataFrame, outfile: Path) -> None:
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_covariates"], as_index=False)["coverage"].mean()
    line_plot(agg, x="n_covariates", title="Coverage vs dimensionality (rho=0.2)", outfile=outfile)


def plot_ci_length_by_scenario(df: pd.DataFrame, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values="ci_length_mean")
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Avg CI length")
    ax.set_title("Average CI length across scenarios")
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    ax.legend(title="Configuration", frameon=False)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def plot_ci_length_boxplots(df_runs: pd.DataFrame, outdir: Path) -> None:
    """
    Save one boxplot per scenario to keep figures readable.
    """
    if df_runs.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in _ordered_scenarios(df_runs["scenario"].unique().tolist()):
        subset = df_runs[df_runs["scenario"] == scenario]
        if subset.empty:
            continue
        labels = _ordered_labels(subset["label"].unique().tolist())
        data = [subset[subset["label"] == label]["ci_length"].dropna() for label in labels]
        colors = [COLORS.get(label, None) for label in labels]
        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color if color else "#cccccc")
        ax.set_title(f"CI length distribution: {scenario}")
        ax.set_ylabel("CI length")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / f"ci_length_boxplot_{scenario}.png", dpi=300)
        plt.close(fig)


def plot_beta_hat_distribution(df_runs: pd.DataFrame, outdir: Path, beta_true: float = 2.0) -> None:
    """
    Histogram of treatment_effect_hat with Gaussian overlay and true beta line, one file per scenario.
    """
    if df_runs.empty or "treatment_effect_hat" not in df_runs.columns:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in _ordered_scenarios(df_runs["scenario"].unique().tolist()):
        subset = df_runs[df_runs["scenario"] == scenario]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = subset["treatment_effect_hat"].dropna()
        if all_vals.empty:
            plt.close(fig)
            continue
        x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
        for label in _ordered_labels(subset["label"].unique().tolist()):
            group = subset[subset["label"] == label]
            vals = group["treatment_effect_hat"].dropna()
            if vals.empty:
                continue
            color = COLORS.get(label, None)
            ax.hist(vals, bins=30, density=True, alpha=0.4, label=f"{label} hist", color=color)
            mu, sigma = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if sigma > 0:
                from scipy.stats import norm
                ax.plot(x_grid, norm.pdf(x_grid, loc=mu, scale=sigma),
                        color=color, linewidth=2, label=f"{label} N({mu:.2f}, {sigma:.2f})")
        ax.axvline(beta_true, color="black", linestyle="--", linewidth=1, label="True treatment effect")
        ax.set_title(f"Sampling distribution of treatment_effect_hat: {scenario}")
        ax.set_xlabel("treatment_effect_hat")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / f"treatment_effect_hat_distribution_{scenario}.png", dpi=300)
        plt.close(fig)


def plot_selection_box_violin(df_runs: pd.DataFrame, metric: str, outdir: Path) -> None:
    """
    Boxplot and violin plot for selected-controls counts per scenario and configuration (one file per scenario+metric).
    """
    if df_runs.empty or metric not in df_runs.columns:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in _ordered_scenarios(df_runs["scenario"].unique().tolist()):
        subset = df_runs[df_runs["scenario"] == scenario]
        if subset.empty:
            continue
        labels = _ordered_labels(subset["label"].unique().tolist())
        data = [subset[subset["label"] == label][metric].dropna() for label in labels]
        colors = [COLORS.get(label, None) for label in labels]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # Boxplot
        bp = axes[0].boxplot(data, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color if color else "#cccccc")
        axes[0].set_title(f"{metric} boxplot: {scenario}")
        axes[0].set_ylabel(metric)
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        # Violin plot
        vp = axes[1].violinplot(data, showmeans=True, showmedians=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color if color else "#cccccc")
            body.set_alpha(0.6)
        axes[1].set_title(f"{metric} violin: {scenario}")
        axes[1].set_xticks(range(1, len(labels) + 1))
        axes[1].set_xticklabels(labels, rotation=15)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(outdir / f"{metric}_{scenario}.png", dpi=300)
        plt.close(fig)


def plot_k_line(df: pd.DataFrame, metric: str, ylabel: str, title: str, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in _ordered_labels(df["label"].unique().tolist()):
        group = df[df["label"] == label]
        ordered_vals = [group[group["scenario"] == sc][metric].values[0] for sc in order if sc in set(group["scenario"])]
        ax.plot(x[:len(ordered_vals)], ordered_vals, marker="o", label=label, color=COLORS.get(label, None))
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=30)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_summary_dashboard(df: pd.DataFrame, outfile: Path) -> None:
    """
    Combined dashboard: coverage, bias, CI length, and selected-controls counts across scenarios.
    """
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    metrics = [
        ("coverage", "Coverage"),
        ("bias", "Bias"),
        ("ci_length_mean", "Avg CI length"),
        ("n_selected_outcome_controls_mean", "Avg outcome controls"),
        ("n_selected_treatment_controls_mean", "Avg treatment controls"),
    ]
    n_axes = len(metrics)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_flat = axes.flatten()

    for ax, (metric, title) in zip(axes_flat, metrics):
        pivot = df.pivot(index="scenario", columns="label", values=metric)
        if pivot.empty:
            ax.axis("off")
            continue
        pivot = pivot.loc[order]
        pivot = pivot[_ordered_labels(pivot.columns.tolist())]
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, None) for c in pivot.columns])
        if metric == "coverage":
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(metric)
        if ax is axes_flat[0]:
            ax.legend(title="Configuration", frameon=False, fontsize=8)
        else:
            ax.legend().remove()

    # Hide any unused axes
    for ax in axes_flat[n_axes:]:
        ax.axis("off")

    fig.suptitle("Double LASSO performance summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_overall_summary(df: pd.DataFrame, outfile: Path) -> None:
    """
    Aggregate performance across all scenarios (per configuration).
    """
    metrics = [
        ("coverage", "Coverage"),
        ("bias", "Bias"),
        ("rmse", "RMSE"),
        ("ci_length_mean", "Avg CI length"),
        ("n_selected_outcome_controls_mean", "Avg outcome controls"),
        ("n_selected_treatment_controls_mean", "Avg treatment controls"),
    ]
    grouped = df.groupby("label").agg({m[0]: "mean" for m in metrics}).reset_index()
    grouped["label"] = pd.Categorical(grouped["label"], categories=LABEL_ORDER, ordered=True)
    grouped = grouped.sort_values("label")
    n_axes = len(metrics)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes_flat = axes.flatten()

    for ax, (metric, title) in zip(axes_flat, metrics):
        subset = grouped[["label", metric]].set_index("label")
        subset.plot(kind="bar", ax=ax, legend=False, color=[COLORS.get(lbl, "#888888") for lbl in subset.index])
        if metric == "coverage":
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, ha="right")
        ax.set_ylabel(metric)

    for ax in axes_flat[n_axes:]:
        ax.axis("off")

    fig.suptitle("Overall performance summary (averaged across scenarios)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries()
    run_data = load_run_data()

    plot_coverage_by_scenario(summaries, PLOTS_DIR / "coverage_by_scenario.png")
    plot_metric_by_scenario(
        summaries,
        metric="bias",
        ylabel="Bias",
        title="Bias across scenarios",
        outfile=PLOTS_DIR / "bias_by_scenario.png",
    )
    plot_metric_by_scenario(
        summaries,
        metric="rmse",
        ylabel="RMSE",
        title="RMSE across scenarios",
        outfile=PLOTS_DIR / "rmse_by_scenario.png",
    )
    plot_coverage_vs_rho(summaries, PLOTS_DIR / "coverage_vs_rho.png")
    plot_coverage_vs_n(summaries, PLOTS_DIR / "coverage_vs_n.png")
    plot_coverage_vs_p(summaries, PLOTS_DIR / "coverage_vs_p.png")
    plot_ci_length_by_scenario(summaries, PLOTS_DIR / "ci_length_by_scenario.png")
    plot_ci_length_boxplots(run_data, PLOTS_DIR)
    plot_beta_hat_distribution(run_data, PLOTS_DIR)
    plot_selection_box_violin(run_data, metric="n_selected_outcome_controls", outdir=PLOTS_DIR)
    plot_selection_box_violin(run_data, metric="n_selected_treatment_controls", outdir=PLOTS_DIR)
    plot_k_line(
        summaries,
        metric="n_selected_outcome_controls_mean",
        ylabel="Average selected outcome controls",
        title="Selected controls (outcome model) across scenarios",
        outfile=PLOTS_DIR / "n_selected_outcome_controls_line.png",
    )
    plot_k_line(
        summaries,
        metric="n_selected_treatment_controls_mean",
        ylabel="Average selected treatment controls",
        title="Selected controls (treatment model) across scenarios",
        outfile=PLOTS_DIR / "n_selected_treatment_controls_line.png",
    )
    plot_summary_dashboard(summaries, PLOTS_DIR / "summary_dashboard.png")
    plot_overall_summary(summaries, PLOTS_DIR / "summary_overall.png")


if __name__ == "__main__":
    main()
