# Double LASSO simulation toolkit

Monte Carlo framework to study coverage and interval length for Double LASSO in high-dimensional linear models. Supports plug-in penalties from Chernozhukov et al. and cross-validated penalties, with batch scenario runs and sweeps.

## Project layout
- `dgps/`: data-generating processes (e.g., `static.py`)
- `estimators/lasso.py`: Double LASSO estimator, plug-in and CV alphas
- `runner.py`: Monte Carlo loop wrapper
- `orchestrator.py`: parameter sweeps and plotting
- `scenarios.py`: predefined scenario grid
- `main.py`: CLI entry point

## Requirements
- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`, `scipy`

## Running simulations
Single design (plug-in penalty by default):
```bash
python3 main.py run --R 200 --n 200 --p 100 --s 5 --beta1 2.0 --rho 0.2 --out results.csv
```

Use cross-validated penalties instead of plug-in:
```bash
python3 main.py run --R 200 --n 200 --p 100 --s 5 --beta1 2.0 --rho 0.2 --use_cv --out results.csv
```

Sweep one parameter (example: sample size):
```bash
python3 main.py sweep --param n --values 80,120,200,320 --R 100 --out sweep_results.csv
```

Run the full scenario grid:
```bash
python3 main.py scenarios --outdir results
```

Each CSV includes `beta1_hat`, `se_HC3`, `ci_low`, `ci_high`, `ci_length`, `covered`, `k_y`, `k_d`, and the alphas used (`alpha_y`, `alpha_d`).

## Penalty choices
- **Plug-in** (default): iterative sigma-based rule from Chernozhukov et al., controlled by `plugin_c` and `plugin_alpha_level` parameters inside `double_lasso_ci`.
- **Cross-validation**: enable with `--use_cv` to select alpha separately for Y and D via `LassoCV`.

## Notes
- Coverage is computed with HC3 robust standard errors on the residual-on-residual regression.
- Results are written under `results/` (plug-in) or any directory you pass; CV runs can be separated using `results_cv/` if desired.
