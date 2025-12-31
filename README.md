# Evaluating the Confidence-Interval Performance of the Double LASSO Estimator in High-Dimensional Linear Models

This project studies the finite-sample behavior of the Double LASSO estimator for inference in high-dimensional linear regression models. Using controlled Monte Carlo simulations, we examine how confidence-interval coverage and length behave under different data-generating processes (Gaussian, heavy-tailed, and approximate-sparse) and varying levels of dimensionality and covariate correlation. The simulations assess how regularization choices—plug-in versus cross-validated penalties—affect the stability and reliability of post-selection inference relative to ordinary least squares in settings where covariates are numerous relative to sample size. 

## Project structure
- Simulation scenarios: structured configurations controlling sample size, dimensionality, sparsity, correlation, and noise.
- DGPs: light-tailed and heavy-tailed designs.
- Estimators: Double LASSO (alternative penalty rules) and OLS for comparison.
- Outputs: summary statistics from repeated Monte Carlo replications, saved as CSV files for downstream analysis and plotting.

## Installation
Install the required Python dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Reproducibility
Seeds are fixed at the scenario level and can be overridden via CLI flags. Given the same seed, code version, and environment, results are deterministic.

## Difference Between static.py and static_easier.py DGPs

The two data-generating processes differ primarily in how demanding they make the variable-selection and partialling-out steps that underpin Double LASSO. 

The original static.py design is deliberately difficult and meant to resemble realistic empirical environments. In this setting, the same covariates play an important role in both the treatment and outcome equations, generating strong confounding and tightly intertwined signals. Covariates are correlated, relevant effects are not always dominant, and some confounders exert only modest marginal influence. This makes LASSO selection inherently fragile in finite samples, so small errors in the first-stage nuisance estimation can translate into noticeable variability in the final estimate, even though Neyman orthogonality continues to protect against first-order bias asymptotically. 

By contrast, static_easier.py relaxes these difficulties in a controlled way. Confounding is weaker, the signals driving the treatment and outcome are more cleanly separated, and the effective sparsity structure is easier to detect. As a result, the nuisance functions are simpler to learn, LASSO selection is more stable, and the partialling-out step behaves closer to its idealized theoretical counterpart. The easier design therefore serves as a transparent benchmark that illustrates the method’s intended behavior, while the original design functions as a stress test that reveals how Double LASSO performs when selection and confounding are genuinely challenging.

## Running the simulations

Optional flags let you switch estimation methods or design features (e.g., `--use_cv` for cross-validated penalties, `--dgp heavy_tail`, or `--estimator ols`).

### Run with different DGPs
Choose a data-generating process with `--dgp` (options: `static`, `static_easier`, `heavy_tail`). Use separate output folders to avoid overwriting.

```bash
python3 main.py scenarios --dgp static --outdir results

python3 main.py scenarios --dgp static_easier --outdir results_easierdgp

python3 main.py scenarios --dgp heavy_tail --outdir results_heavy
```
You can also combine with cross-validated penalties:
```bash
python3 main.py scenarios --dgp static --use_cv --outdir results_cv
python3 main.py scenarios --dgp static_easier --use_cv --outdir results_ecv
python3 main.py scenarios --dgp heavy_tail --use_cv --outdir results_heavycv
```

### Generate plots from simulation outputs
```bash
python3 plots.py
```
This expects summary CSV files produced by prior simulation runs under the results directories.

## Authors
Shokhrukhkhon Nishonkulov  
Olimjon Umurzokov  
Damir Abdulazizov  
M.Sc. Economics, University of Bonn  
Research Module in Econometrics and Statistics (2025)  
Professor: Vladislav Morozov
