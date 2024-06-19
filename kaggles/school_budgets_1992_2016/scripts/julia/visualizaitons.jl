# scripts/julia/visualizations.jl

using Plots
using DataFrames
using CSV
using PyCall

# Load data
districts_clean = CSV.read("data/processed/districts_clean.csv", DataFrame)

# Simple scatter plot
scatter(districts_clean.student_count, districts_clean.log_total_exp,
    xlabel="Student Count", ylabel="Log Total Expenditure",
    title="Student Count vs. Log Total Expenditure")
savefig("reports/figures/student_count_vs_log_total_expenditure.png")

# Use PyCall to call Python functions (if needed)
py"""
import joblib
params = joblib.load('models/jax/regression_model.pkl')
print(params)
"""

# More complex visualizations...
