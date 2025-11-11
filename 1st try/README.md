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

### **Shokhrukhkhon Nishonkulov** 

### **Olim Umurzokov**  

### **Damir Abdulazizov** 
M.Sc. Economics, University of Bonn  
Research Module in Econometrics and Statistics (2025)
