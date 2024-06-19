# run_all.sh

# activate conda env
source activate school_budgets_1992_2016

Rscript scripts/R/preprocessing_school_budgets.R

# Run python models
python scripts/python/train_model.py
python scrips/jax/optimization.py

# run julia visuals
julia scripts/julia/visualizations.jl

#deactivate env
source deactivate
