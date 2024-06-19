# Adaptive Analysis Framework for School Budgets

This project focuses on leveraging legacy and emerging statistical software frameworks to build a robust Linear Mixed Model (LMM) analysis. The primary goal is to enable meaningful decision-making based on data and learning heuristics.

## Project Structure
school_budgets/
├── data/
│ ├── raw/ # Raw data files
│ ├── processed/ # Processed data files
│ └── external/ # External data sources
├── scripts/
│ ├── R/ # R scripts for data preprocessing and analysis
│ ├── python/ # Python scripts for ML and visualization
│ ├── julia/ # Julia scripts for advanced modeling
│ ├── jax/ # JAX, EvoJAX, and NeuroJAX scripts
│ └── others/ # Other miscellaneous scripts
├── notebooks/
│ ├── R/ # R notebooks
│ ├── python/ # Python notebooks
│ └── julia/ # Julia notebooks
├── models/
│ ├── r/ # Saved R models
│ ├── pytorch/ # Saved PyTorch models
│ ├── jax/ # Saved JAX models
│ └── julia/ # Saved Julia models
├── reports/
│ ├── figures/ # Generated figures and plots
│ └── papers/ # Papers and reports
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment file
├── README.md # Project overview
└── LICENSE # License information


## Project Goals

- **Data Preprocessing (R):** Clean and prepare datasets using R scripts.
- **Statistical Analysis (R):** Perform LMM analysis and generate visualizations.
- **Advanced Modeling (Julia):** Use Julia for computationally intensive tasks.
- **Machine Learning (Python + PyTorch):** Train and evaluate models.
- **Neural Network Training (JAX):** Leverage JAX, EvoJAX, and NeuroJAX.
- **Visualization (Julia + R + Python + Julia (for advanced ML):** Create detailed plots and charts.
- **Reporting:** Compile results into comprehensive reports.

## How to Use

1. **Create and Activate a Conda Environment:**
    ```sh
    conda create -f environment.yml
    conda activate school_budgets
    ```

2. **Run Preprocessing Script (R):**
    ```sh
    Rscript scripts/R/preprocessing_school_budgets.R
    ```

3. **Run Machine Learning Script (Python):**
    ```sh
    python scripts/python/train_model.py
    ```

4. **Run Advanced Modeling Script (Julia):**
    ```sh
    julia scripts/julia/advanced_modeling.jl
    ```

5. **Generate Visualizations (Python and R):**
    ```sh
    python scripts/python/visualize_data.py
    Rscript scripts/R/visualize_data.R
    ```

6. **Push to GitHub:**
    ```sh
    git add .
    git commit -m "Added new analysis and models"
    git push origin adaptive-analysis-framework
    ```

## Contributions

- Follow the structure and coding guidelines.
- Document your work and provide meaningful commit messages.
- Engage in discussions and reviews to improve the project.

## License

This project is licensed under the MIT License.
