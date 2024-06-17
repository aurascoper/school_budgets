# Install necessary libraries if not already installed
if (!requireNamespace("tidyr", quietly = TRUE)) {
  install.packages("tidyr")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}

# Load necessary libraries
library(dplyr)
library(tidyr)
library(lme4)
library(data.table)
library(ggplot2)

# Load dataset
districts <- fread("districts.csv")
states <- fread("states.csv")
naep <- fread("naep.csv")

# Data preprocessing
data_clean <- data %>%
  filter(!is.na(total_expenditure)) %>%
  mutate(log_expenditure = log(total_expenditure))

write.csv(data_clean, "preprocessed_school_budgets.csv", row.names = FALSE)

# Fit the Linear Mixed Model
lmm <- lmer(log_expenditure ~ student_count + (1 | school), data = data_clean)

# Summary of the model
print(summary(lmm))

# Save model output
saveRDS(lmm, "lmm_school_budgets_model.rds")

# Visualization
# Histogram of log_expenditure
ggplot(data_clean, aes(x = log_expenditure)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  labs(title = "Distribution of Log Expenditure", x = "Log Expenditure", y = "Frequency") +
  theme_minimal()

# Boxplot of log_expenditure by school
ggplot(data_clean, aes(x = school, y = log_expenditure)) +
  geom_boxplot() +
  labs(title = "Log Expenditure by School", x = "School", y = "Log Expenditure") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme_minimal()

save

# Scatter plot of student_count vs log_expenditure
ggplot(data_clean, aes(x = student_count, y = log_expenditure)) +
  geom_point(color = "blue") +
  labs(title = "Student Count vs Log Expenditure", x = "Student Count", y = "Log Expenditure") +
  theme_minimal()

# Save the LMM model
saveRDS(lmm, "lmm_school_budgets_model.rds")
