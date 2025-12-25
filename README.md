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

## Running the simulations
### Run all predefined scenarios
```bash
python3 main.py scenarios --outdir results
```
Optional flags let you switch estimation methods or design features (e.g., `--use_cv` for cross-validated penalties, `--dgp heavy_tail`, or `--estimator ols`).

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
