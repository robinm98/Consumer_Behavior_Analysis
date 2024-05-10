#### MLBA Project - Logistic Regression ####
############################################

# Load the required libraries
library(dplyr)
library(here)
library(tidyverse)
library(fastDummies)
library(dplyr)
library(ggplot2)
library(lattice)
library(caret)
library(pROC)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))


#### Logistic Regression #####
##############################

### Preprocessing ###

# Convert categorical variables to factor
categorical_vars <- c("sales_channel", "trip_type", "flight_day", "continent")
data <- data |> 
  mutate(across(all_of(categorical_vars), as.factor))

# Prepare dummy variables
data <- data |> 
  dummy_cols(select_columns = categorical_vars, remove_first_dummy = TRUE)

# Remove unneeded columns if they are not used in any models (verify first), do we want to keep wants preferred seat and meal?
data_lr1 <- data |> 
  dplyr::select(-route, -booking_origin, -departure, -arrival, -flight_day, -continent, -sales_channel, -trip_type)


# Splitting data into training and testing sets
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(data_lr1$wants_extra_baggage, p = 0.8, list = FALSE)
train_data <- data_lr1[trainIndex, ]
test_data <- data_lr1[-trainIndex, ]

### wants_extra_baggage ###
###########################

### Fit the initial logistic regression model ###

# Initial logistic regression model
logist_model1 <- glm(wants_extra_baggage ~ ., data = train_data, family = "binomial")

# Stepwise backward elimination based on AIC
reduced_model <- stepAIC(logist_model1, direction = "backward")
summary(reduced_model)

# Capture the formula of the reduced model
reduced_formula <- formula(reduced_model)
reduced_formula

# Convert 'wants_extra_baggage' from numeric to factor
train_data$wants_extra_baggage <- factor(train_data$wants_extra_baggage, levels = c(0, 1), labels = c("No", "Yes"))

# Convert 'wants_extra_baggage' from numeric to factor
test_data$wants_extra_baggage <- factor(test_data$wants_extra_baggage, levels = c(0, 1), labels = c("No", "Yes"))

### Cross-validation ###

# Setup cross-validation on training data
train_control <- trainControl(
  method = "cv",
  number = 10,  # number of folds in cross-validation
  savePredictions = "final",
  classProbs = TRUE  # since it's a classification problem
)


# Train the model using cross-validation
cv_model <- train(
  reduced_formula,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Model summary
summary(cv_model)
print(cv_model)


# Predict on the test data
test_prob <- predict(cv_model, newdata = test_data, type = "prob")
test_pred <- ifelse(test_prob[, "Yes"] > 0.681, "Yes", "No")


test_pred <- as.factor(test_pred)
test_data$wants_extra_baggage <- as.factor(test_data$wants_extra_baggage)

# Confusion matrix
confusionMatrix(data=test_pred, reference = test_data$wants_extra_baggage)

# Boxplot of the probabilities
boxplot(test_prob[, "Yes"]~test_data$wants_extra_baggage, col = c("red", "blue"), xlab = "Wants Extra Baggage", ylab = "Probability of Yes", main = "Boxplot of Predicted Probabilities")

# Average predicted probability for those who did NOT want extra baggage
mean_yes_given_no <- mean(test_prob[test_data$wants_extra_baggage == "No", "Yes"])

# Average predicted probability for those who DID want extra baggage
mean_yes_given_yes <- mean(test_prob[test_data$wants_extra_baggage == "Yes", "Yes"])

# Output these averages
cat("Average Probability of Yes given Actual No: ", mean_yes_given_no, "\n")
cat("Average Probability of Yes given Actual Yes: ", mean_yes_given_yes, "\n")


# ROC curve
roc_curve <- roc(test_data$wants_extra_baggage, test_prob[, "Yes"])
plot(roc_curve, print.thres="best", col = "blue", lwd = 2, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")


# AUC
auc(roc_curve)


