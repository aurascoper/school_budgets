# scripts/python/train_model.py

import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.scipy as jsp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Load preprocessed data
districts_clean = pd.read_csv('data/processed/districts_clean.csv')
states_clean = pd.read_csv('data/processed/states_clean.csv')
naep_clean = pd.read_csv('data/processed/naep_clean.csv')

# Example regression model
X = districts_clean[['student_count']]
y = districts_clean['log_total_exp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model(X, params):
    return jnp.dot(X, params['weights']) + params['bias']

def loss(params, X, y):
    preds = model(X, params)
    return jnp.mean((preds - y) ** 2)

params = {'weights': jnp.ones(X_train.shape[1]), 'bias': 0.0}

# Gradient descent step
@jit
def step(params, X, y, lr=0.01):
    grads = grad(loss)(params, X, y)
    return {'weights': params['weights'] - lr * grads['weights'],
            'bias': params['bias'] - lr * grads['bias']}

# Training loop
for epoch in range(1000):
    params = step(params, X_train.values, y_train.values)

# Save the model
joblib.dump(params, 'models/jax/regression_model.pkl')

# Evaluate the model
y_pred = model(X_test.values, params)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
