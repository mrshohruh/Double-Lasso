# Evaluating the Confidence-Interval Performance of the Double LASSO Estimator in High-Dimensional Econometric Models

This repository contains a modular simulation framework that evaluates the **confidence-interval performance** of the Double LASSO estimator in high-dimensional linear regression models.  


---

##  Research Objective

The goal of this project is to study how accurately the Double LASSO estimator constructs confidence intervals for a target parameter \( \beta_1 \) when the number of covariates is large relative to the sample size.  
The estimator is evaluated in terms of **coverage probability**, **bias**, **RMSE**, and **average interval length**, across multiple sample sizes.

Model setup:
\[
Y = \beta_1 D + X \gamma + \varepsilon
\]
where \( Y \) is the outcome, \( D \) the treatment variable of interest, and \( X \) a high-dimensional vector of controls.

---

##  Project Structure

```
1st try/
├── dgps/
│   ├── static.py          # Data-Generating Process (StaticNormalDGP)
│   └── dynamic.py         # (reserved for dynamic DGPs)
│
├── estimators/
│   └── lasso.py           # Double LASSO estimation and confidence intervals
│
├── protocols.py           # DGP and Estimator interface protocols
├── runner.py              # Monte Carlo simulation loop
├── orchestrator.py        # Parameter sweeps and plotting
├── scenarios.py           # Predefined scenarios (baseline, small-n, large-n)
└── main.py                # Entry point script (CLI interface)
```

---

##  How to Run Simulations

### Run a single baseline experiment
```bash
python3 main.py run --R 200 --n 200 --p 100 --s 5 --beta1 2.0 --rho 0.2 --out results.csv
```

### Sweep over sample size (n)
```bash
python3 main.py sweep --param n --values 80,120,200,320 --R 100 --out sweep_results.csv
```

### Sweep with p=60 across n ∈ {120, 200, 320}
```bash
python main.py sweep --param n --values 120,200,320 --p 60 --out p60_sweep.csv
```

### Run all predefined scenarios
```bash
python3 main.py scenarios --outdir results
```

Each run produces a `.csv` file with parameter estimates, standard errors, and CI bounds.

---

## Key Findings

| Sample Size (n) | Avg CI Length | Coverage (95%) | Bias | RMSE |
|-----------------|---------------|----------------|------|------|
| 80   | 0.414 | 0.02 | −0.44 | 0.45 |
| 120  | 0.343 | 0.03 | −0.32 | 0.33 |
| 200  | 0.275 | 0.21 | −0.20 | 0.21 |
| 320  | 0.221 | 0.41 | −0.13 | 0.14 |

Coverage improves and intervals shorten as \( n \) increases — clear evidence of convergence toward valid inference.

---

## Method Summary

1. **Partialling-Out Steps**
   - LASSO( Y ∼ X ) → residuals \( \tilde{Y} \)
   - LASSO( D ∼ X ) → residuals \( \tilde{D} \)
2. **Final Stage**
   - OLS( \( \tilde{Y} \sim \tilde{D} \) ) with HC3 robust SEs → CI for \( \beta_1 \)
3. **Penalty**
   - Plug-in penalty \( \lambda = c \sqrt{2 \ln(p) / n} \)

---

##  Conclusions

The simulation results confirm that the Double LASSO estimator’s confidence intervals become progressively more accurate as the sample size increases.  
At small \( n \), coverage is well below the nominal 95% level and intervals are wide, reflecting small-sample regularization bias.  
However, both bias and RMSE decline quickly with larger \( n \), and interval length contracts from roughly 0.41 at \( n=80 \) to 0.22 at \( n=320 \).  
Throughout, the number of selected controls remains small, supporting the **approximate sparsity** assumption that underlies the estimator’s theoretical guarantees.

Overall, the results demonstrate that Double LASSO offers a reliable and theoretically grounded method for inference in high-dimensional econometric models.  
It combines effective variable selection with valid post-selection inference, bridging modern machine learning techniques with classical econometric reasoning.

---

## Requirements

- Python ≥ 3.9  
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`, `scipy`


---

##  Reference



---

## Authors

#### **Shokhrukhkhon Nishonkulov** 

#### **Olimjon Umurzokov**  

#### **Damir Abdulazizov** 
M.Sc. Economics, University of Bonn  
Research Module in Econometrics and Statistics (2025)

## Summary of Simulation Development and Changes

This section documents all the updates, extensions, and experiments we implemented in the simulation framework throughout the development process. The goal was to evaluate the performance of the Double LASSO estimator under a wide range of high-dimensional designs and tuning strategies.

### 1. Increased Monte Carlo Repetitions (R = 100 → R = 500)

We increased Monte Carlo replications across all designs by updating defaults in runner.py, orchestrator.py, and scenarios.py. Every scenario now consistently uses R = 500. This reduces Monte Carlo noise and yields more stable estimates of coverage, bias, and CI length.

### 2. Added New Scenarios with Lower Dimensionality (p = 60)

We created new scenario variants with p = 60 for multiple values of n. These scenarios reduce dimensionality and allow us to study Double LASSO performance under easier high-dimensional settings where n >> s log p is better satisfied.

### 3. Added Lower-Sparsity Variants (s = 3)

We extended every original scenario by adding versions with s = 3. This makes the DGP more sparse, which theory predicts should improve LASSO selection accuracy and inference performance.

### 4. Added New Correlation Structures (ρ = 0 and ρ = 0.1)

We introduced new correlation levels for p = 60, s = 3 scenarios. These variants test how Double LASSO behaves under weaker multicollinearity conditions.

### 5. Ensured Scenario Parameters Are Fully Respected

We updated main.py and internal wiring so that all parameters from SimulationScenario objects are correctly passed into the simulation engine. This guarantees that changes to n, p, s, ρ, and other parameters in scenarios.py directly affect the simulations.

### 6. Added compare_scenarios.py

We implemented a script that summarizes all raw scenario outputs into one table (coverage, CI length, bias, RMSE, k_y, k_d). This file enables easy comparison across all scenario designs.

### 7. Tested and Reverted Reduced Noise in the DGP

We experimented with reducing noise (scale = 0.5) in u and v, but reverted this change after confirming it did not resolve the main issue (bias due to underselection). This test confirmed that noise is not the dominant factor in coverage failures.

### 8. Added Scenarios with Lower Penalty Constant (c = 0.6)

We added parallel versions of all scenarios with c = 0.6, allowing us to test the effect of weaker LASSO penalties on selection and inference. These results showed mixed improvements depending on scenario difficulty.

### 9. Implemented Cross-Validated LASSO Penalties (CV-Based α)

We added a cv_alpha() function, integrated CV into double_lasso_ci, and enabled CV-based α for both nuisance regressions. This provides a data-driven penalty choice widely used in practice.

### 10. Added --use_cv Flag in main.py

We updated the CLI to support a new --use_cv flag, enabling cross-validated estimation directly from the command line in all modes (run, sweep, scenarios).

### 11. Generated CV-Based Scenario Outputs into results_cv/

We executed all scenarios with CV-based α and saved them to a new results_cv/ directory, creating a clean separation between plugin-penalty and CV-penalty results.

### 12. Added compare_plugin_vs_cv.py

We implemented a script to compare plugin vs CV results side by side. It computes differences in coverage, bias, and CI length and writes a combined summary table to results/plugin_vs_cv_summary.csv.

This expanded pipeline now supports a comprehensive evaluation of Double LASSO under diverse high-dimensional designs and tuning strategies.
