# Install necessary packages if not already installed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

packages <- c("tidyr", "dplyr", "lme4", "car", "ggplot2", "data.table")
lapply(packages, install_if_missing)

# Load libraries
library(dplyr)
library(tidyr)
library(lme4)
library(car)
library(ggplot2)
library(data.table)

# Load datasets and check for invalid values
check_na_inf <- function(data) {
  invalid_values <- sapply(data, function(x) sum(is.na(x) | is.nan(x) | is.infinite(x)))
  print("Count of NA/NaN/Inf values in each column:")
  print(invalid_values)
  return(invalid_values)
}

districts <- fread("districts.csv")
states <- fread("states.csv")
naep <- fread("naep.csv")

print("First few rows of districts:")
print(head(districts))

print("First few rows of states:")
print(head(states))

print("First few rows of naep:")
print(head(naep))

print("Checking districts dataset:")
check_na_inf(districts)

print("Checking states dataset:")
check_na_inf(states)

print("Checking naep dataset:")
check_na_inf(naep)

# Preprocess data
preprocess_districts <- function(data) {
  data <- data[!is.na(TOTALREV)]
  numeric_cols <- c("TOTALEXP", "TFEDREV", "TSTREV", "TLOCREV", "TCURINST", "TCURSSVC", "TCURONON", "TCAPOUT")
  for (col in numeric_cols) {
    data[[col]] <- as.numeric(data[[col]])
    if (any(is.na(data[[col]]))) {
      print(paste("Coercion issue detected in column:", col))
      data <- data[!is.na(data[[col]])]
    }
  }
  data[, `:=`(log_total_exp = log(TOTALEXP),
              log_tfedrev = log(TFEDREV),
              log_tstrev = log(TSTREV),
              log_tlocrev = log(TLOCREV),
              log_tcurinst = log(TCURINST),
              log_tcurssvc = log(TCURSSVC),
              log_tcuronon = log(TCURONON),
              log_tcapout = log(TCAPOUT))]
  setnames(data, "ENROLL", "student_count")
  setnames(data, "NAME", "school")
  return(data)
}

preprocess_states <- function(data) {
  data <- data[!is.na(TOTAL_REVENUE)]
  numeric_cols <- c("TOTAL_EXPENDITURE", "FEDERAL_REVENUE", "STATE_REVENUE", "LOCAL_REVENUE", "INSTRUCTION_EXPENDITURE", "SUPPORT_SERVICES_EXPENDITURE", "OTHER_EXPENDITURE", "CAPITAL_OUTLAY_EXPENDITURE")
  for (col in numeric_cols) {
    data[[col]] <- as.numeric(data[[col]])
    if (any(is.na(data[[col]]))) {
      print(paste("Coercion issue detected in column:", col))
      data <- data[!is.na(data[[col]])]
    }
  }
  data[, `:=`(log_total_exp = log(TOTAL_EXPENDITURE),
              log_federal_rev = log(FEDERAL_REVENUE),
              log_state_rev = log(STATE_REVENUE),
              log_local_rev = log(LOCAL_REVENUE),
              log_instruction_exp = log(INSTRUCTION_EXPENDITURE),
              log_support_services_exp = log(SUPPORT_SERVICES_EXPENDITURE),
              log_other_exp = log(OTHER_EXPENDITURE),
              log_capital_outlay_exp = log(CAPITAL_OUTLAY_EXPENDITURE))]
  return(data)
}

preprocess_naep <- function(data) {
  data <- data[!is.na(AVG_SCORE)]
  data[, `:=`(
    year = as.integer(YEAR),
    avg_score = as.numeric(AVG_SCORE),
    test_subject = as.factor(TEST_SUBJECT),
    test_year = as.integer(TEST_YEAR)
  )]
  numeric_cols <- c("year", "avg_score", "test_year")
  for (col in numeric_cols) {
    if (any(is.na(data[[col]]))) {
      print(paste("Coercion issue detected in column:", col))
      data <- data[!is.na(data[[col]])]
    }
  }
  return(data)
}

districts_clean <- preprocess_districts(districts)
states_clean <- preprocess_states(states)
naep_clean <- preprocess_naep(naep)

print("Checking districts_clean dataset:")
check_na_inf(districts_clean)

print("Checking states_clean dataset:")
check_na_inf(states_clean)

print("Checking naep_clean dataset:")
check_na_inf(naep_clean)

# Ensure data integrity and fit Linear Mixed Model (LMM)
data_clean <- districts_clean %>%
  rename(log_expenditure = log_total_exp) %>%
  select(log_expenditure, student_count, school) %>%
  filter(!is.na(log_expenditure) & !is.na(student_count) & !is.na(school)) %>%
  mutate(student_count_scaled = scale(student_count))

print("First few rows of data_clean:")
print(head(data_clean))

# Check for NA/NaN/Inf values in the cleaned data
check_na_inf(data_clean)

# Remove any remaining rows with NA/NaN/Inf values in log_expenditure
data_clean <- data_clean %>%
  filter(!is.na(log_expenditure) & !is.nan(log_expenditure) & !is.infinite(log_expenditure))

if (nrow(data_clean) == 0) {
  stop("Data clean resulted in an empty data frame. Check preprocessing steps.")
}

lmm <- lmer(log_expenditure ~ student_count_scaled + (1 | school), data = data_clean)
print(summary(lmm))
saveRDS(lmm, "lmm_school_budgets_model.rds")

# Visualizations
plot1 <- ggplot(data_clean, aes(x = log_expenditure)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  labs(title = "Distribution of Log Expenditure", x = "Log Expenditure", y = "Frequency") +
  theme_minimal()
print(plot1)
ggsave("log_expenditure_histogram.png")

plot2 <- ggplot(data_clean, aes(x = school, y = log_expenditure)) +
  geom_boxplot() +
  labs(title = "Log Expenditure by School", x = "School", y = "Log Expenditure") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme_minimal()
print(plot2)
ggsave("log_expenditure_boxplot_by_school.png")

plot3 <- ggplot(data_clean, aes(x = student_count, y = log_expenditure)) +
  geom_point(color = "blue") +
  labs(title = "Student Count vs Log Expenditure", x = "Student Count", y = "Log Expenditure") +
  theme_minimal()
print(plot3)
ggsave("student_count_vs_log_expenditure.png")

plot4 <- ggplot(data_clean, aes(x = student_count_scaled, y = log_expenditure)) +
  geom_point(color = "blue") +
  labs(title = "Scaled Student Count vs Log Expenditure", x = "Scaled Student Count", y = "Log Expenditure") +
  theme_minimal()
print(plot4)
ggsave("scaled_student_count_vs_log_expenditure.png")

residuals <- resid(lmm)
qqnorm(residuals)
qqline(residuals)
ggsave("qq_plot_residuals.png")

fitted_values <- fitted(lmm)
plot5 <- ggplot(data.frame(fitted = fitted_values, residuals = residuals), aes(x = fitted, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals vs Fitted Values", x = "Fitted Values", y = "Residuals") +
  theme_minimal()
print(plot5)
ggsave("residuals_vs_fitted_values.png")