# scripts/R/preprocessing_school_budgets.R

library(dplyr)
library(tidyr)
library(data.table)

# Load datasets
districts <- fread("data/raw/districts.csv")
states <- fread("data/raw/states.csv")
naep <- fread("data/raw/naep.csv")

# Data cleaning and preprocessing
districts_clean <- districts %>%
  filter(!is.na(TOTALREV)) %>%
  mutate(log_total_exp = log(TOTALEXP))

states_clean <- states %>%
  filter(!is.na(TOTAL_REVENUE)) %>%
  mutate(log_total_exp = log(TOTAL_EXPENDITURE))

naep_clean <- naep %>%
  filter(!is.na(AVG_SCORE))

# Save processed data
fwrite(districts_clean, "data/processed/districts_clean.csv")
fwrite(states_clean, "data/processed/states_clean.csv")
fwrite(naep_clean, "data/processed/naep_clean.csv")
