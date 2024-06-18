# School Budgets Analysis

This project demonstrates data preprocessing, linear mixed model analysis, and visualization using a school budgets dataset.

## Steps

1. **Data Preprocessing**: Clean and prepare the dataset using R.
2. **Linear Mixed Model**: Analyze the data using linear mixed models in R.
3. **Visualization**: Generate visualizations to understand the data better.

## Files

- `preprocessing_school_budgets.R`: R script for data preprocessing, model fitting, and visualization.
- `preprocessed_school_budgets.csv`: The preprocessed dataset.
- `lmm_school_budgets_model.rds`: The saved linear mixed model.
- Visualizations:
  - `log_expenditure_histogram.png`
  - `log_expenditure_boxplot_by_school.png`
  - `student_count_vs_log_expenditure.png`
  - `scaled_student_count_vs_log_expenditure.png`
  - `qq_plot_residuals.png`
  - `residuals_vs_fitted_values.png`

## How to Run

```sh
Rscript preprocessing_school_budgets.R