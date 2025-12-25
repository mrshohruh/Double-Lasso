# Double LASSO simulation toolkit

Monte Carlo framework to study coverage and interval length for Double LASSO in high-dimensional linear models. Supports plug-in penalties from Chernozhukov et al. and cross-validated penalties, with batch scenario runs and sweeps.

The default designs are high-dimensional (p comparable to or exceeding n), so OLS fails and regularization is required.

## Project layout
- `dgps/`: data-generating processes (e.g., `static.py`)
- `estimators/lasso.py`: Double LASSO estimator, plug-in and CV alphas
- `runner.py`: Monte Carlo loop wrapper
- `orchestrator.py`: parameter sweeps and plotting
- `scenarios.py`: predefined scenario grid (small/medium/large with rho variants)
- `main.py`: CLI entry point

## Requirements
- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy`

## Running simulations
Choose a data-generating process with `--dgp static` (default), `--dgp static_easier`, or `--dgp heavy_tail` (Student-t features/errors with df=3 for more outliers).

Single design (plug-in penalty by default):
```bash
python3 main.py run --R 500 --n 200 --p 240 --s 5 --beta1 2.0 --rho 0.2 --out results.csv
```

Use cross-validated penalties instead of plug-in:
```bash
python3 main.py run --R 500 --n 200 --p 240 --s 5 --beta1 2.0 --rho 0.2 --use_cv --out results.csv
```

Heavy-tailed DGP (write to its own folder so you can compare):
```bash
python3 main.py run --R 500 --n 200 --p 240 --s 5 --beta1 2.0 --rho 0.2 --dgp heavy_tail --out results_heavy/results.csv
```

Sweep one parameter (example: sample size):
```bash
python3 main.py sweep --param n --values 120,200,320 --p 320 --R 500 --out sweep_results.csv
```

Run the full scenario grid:
```bash
python3 main.py scenarios --outdir results
```

Classical OLS scenarios (includes low-dim p<n and near p≈n) and write outputs to a dedicated folder:
```bash
python3 main.py scenarios --estimator ols --outdir resultsols
```
To compare against Double LASSO on the same designs, re-run with `--estimator double_lasso` (and optionally a different `--outdir`, e.g., `results_lasso`).
The scenario grid now also includes `p_equals_n` so you can see the exact p=n breakpoint alongside the existing high-dimensional designs.

Run only selected scenarios (comma-separated) without editing `scenarios.py`:
```bash
python3 main.py scenarios --estimator ols --outdir resultsols --scenarios classical_low_dim,p_equals_n,near_p_equals_n
```
Single-scenario example:
```bash
python3 main.py scenarios --estimator ols --outdir resultsols --scenarios small_corr_0_0
```

Cross-validated penalties (store in a separate folder if desired):
```bash
python3 main.py scenarios --outdir results_cv --use_cv
```

Cross-validated penalties with the coverage-friendly DGP (stored under `results_ecv`):
```bash
python3 main.py scenarios --dgp static_easier --use_cv --outdir results_ecv
```


Alternate coverage-friendly DGP (saves to its own folder):
```bash
python3 main.py scenarios --dgp static_easier --outdir results_easierdgp
```

Heavy-tailed scenarios (kept separate for clarity):
```bash
python3 main.py scenarios --dgp heavy_tail --outdir results_heavy
```
If you forget to set `--outdir` for `heavy_tail`, the CLI will automatically redirect scenario outputs to `results_heavy/` instead of mixing them with the Gaussian runs.

Each CSV includes `treatment_effect_hat`, `standard_error_HC3`, `ci_lower`, `ci_upper`, `ci_length`, `covered`, `n_selected_outcome_controls`, `n_selected_treatment_controls`, and the lasso penalties used (`outcome_lasso_penalty`, `treatment_lasso_penalty`).

## Penalty choices
- **Plug-in** (default): iterative sigma-based rule from Chernozhukov et al., controlled by `plugin_c` (default 0.6) and `plugin_alpha_level` parameters inside `double_lasso_ci`.
- **Cross-validation**: enable with `--use_cv` to select alpha separately for Y and D via `LassoCV`.

## Notes
- Coverage is computed with HC3 robust standard errors on the residual-on-residual regression.
- Results are written under `results/` (plug-in) or any directory you pass; CV runs can be separated using `results_cv/` if desired.
