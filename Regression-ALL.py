import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Reproducibility
np.random.seed(7)

# Training data
n_train = 12
x_train = np.sort(np.random.uniform(0, 6, n_train)).reshape(-1, 1)
y_true_train = 0.5 * x_train.squeeze()**3 - 2 * x_train.squeeze()
y_train = y_true_train + np.random.normal(0, 0.3, n_train)

# Test data
x_test = np.linspace(0, 6, 100).reshape(-1, 1)
y_true_test = 0.5 * x_test.squeeze()**3 - 2 * x_test.squeeze()
y_test = y_true_test + np.random.normal(0, 0.3, 100)

degree = 15

# ---- Models (separate pipelines) ----
linear_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scale', StandardScaler()),
    ('model', LinearRegression())
])

lasso_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scale', StandardScaler()),
    ('model', Lasso(alpha=0.01, max_iter=5000))
])

ridge_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1))
])

elastic_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scale', StandardScaler()),
    ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

# Fit models
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
elastic_model.fit(x_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(x_test)
y_pred_lasso = lasso_model.predict(x_test)
y_pred_ridge = ridge_model.predict(x_test)
y_pred_elastic = elastic_model.predict(x_test)

# Metrics
print("Linear R2:", r2_score(y_test, y_pred_linear))
print("Lasso R2:", r2_score(y_test, y_pred_lasso))
print("Ridge R2:", r2_score(y_test, y_pred_ridge))
print("Elastic R2:", r2_score(y_test, y_pred_elastic))

print("Linear MSE:", mean_squared_error(y_test, y_pred_linear))
print("Lasso MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Elastic MSE:", mean_squared_error(y_test, y_pred_elastic))

# ---- Visualization ----
plt.figure(figsize=(14, 5))

# Predictions
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, label="Train Data")
plt.plot(x_test, y_true_test, label="True Function", color='black')
plt.plot(x_test, y_pred_linear, label="Linear")
plt.plot(x_test, y_pred_lasso, label="Lasso")
plt.plot(x_test, y_pred_ridge, label="Ridge")
plt.plot(x_test, y_pred_elastic, label="ElasticNet")
plt.legend()
plt.title("Model Predictions")

# Coefficients
plt.subplot(1, 2, 2)
plt.plot(lasso_model.named_steps['model'].coef_, label="Lasso")
plt.plot(ridge_model.named_steps['model'].coef_, label="Ridge")
plt.plot(elastic_model.named_steps['model'].coef_, label="ElasticNet")
plt.legend()
plt.title("Model Coefficients")

plt.show()