import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression

# 1. Generate data ------------------------------------------------------------
n, p = 200, 100   # n = observations, p = controls
np.random.seed(42)

X = np.random.normal(size=(n, p))
D = X[:, :5].sum(axis=1) + np.random.normal(size=n)  # depends on first 5 controls
Y = 2 * D + X[:, :5].sum(axis=1) + np.random.normal(size=n)  # true beta1 = 2

# 2. Step 1: Lasso for Y ~ X --------------------------------------------------
lasso_y = LassoCV(cv=5).fit(X, Y)
Y_resid = Y - lasso_y.predict(X)

# 3. Step 2: Lasso for D ~ X --------------------------------------------------
lasso_d = LassoCV(cv=5).fit(X, D)
D_resid = D - lasso_d.predict(X)

# 4. Step 3: OLS on residuals -------------------------------------------------
ols = LinearRegression().fit(D_resid.reshape(-1, 1), Y_resid)
beta1_hat = ols.coef_[0]

print(f"Estimated beta1: {beta1_hat:.3f}")
