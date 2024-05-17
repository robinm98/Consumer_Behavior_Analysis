library(readr)
library(here)

# Load the data
NN <- read.csv(here("Data", "NN_results.csv"))
RF <- read.csv(here("Data", "RF_results.csv"))
LR <- read.csv(here("Data", "LR_results.csv"))

# Merge the data and label each model with the algorithm used
all_results <- rbind(NN, RF, LR)
all_results$Model <- factor(rep(c("Neural Network", "Random Forest", "Logistic Regression"), each = nrow(NN)))

# order to have the top accuracy at the top
all_results <- all_results[order(all_results$Accuracy, decreasing = TRUE),]