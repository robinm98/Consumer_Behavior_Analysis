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
library(MASS)
library(ROSE)

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

# Downsampling for wants_extra_baggage
train_data_baggage <- ovun.sample(wants_extra_baggage ~ ., data = train_data, method = "under", N = sum(train_data$wants_extra_baggage == "0") * 2)$data

# Downsampling for wants_preferred_seat
train_data_seat <- ovun.sample(wants_preferred_seat ~ ., data = train_data, method = "under", N = sum(train_data$wants_preferred_seat == "1") * 2)$data

### wants_extra_baggage ###
###########################

### Fit the initial logistic regression model ###

# Initial logistic regression model
logist_model1 <- glm(wants_extra_baggage ~ . - wants_preferred_seat - wants_in_flight_meals, data = train_data_baggage, family = "binomial")

# Stepwise backward elimination based on AIC
reduced_model <- stepAIC(logist_model1, direction = "backward")
summary(reduced_model)

# Capture the formula of the reduced model
reduced_formula <- formula(reduced_model)
reduced_formula

# Convert 'wants_extra_baggage' from numeric to factor for the training set
train_data$wants_extra_baggage <- factor(train_data$wants_extra_baggage, levels = c(0, 1), labels = c("No", "Yes"))

# Convert 'wants_extra_baggage' from numeric to factor for the test set
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

# Output the averages
cat("Average Probability of Yes given Actual No: ", mean_yes_given_no, "\n")
cat("Average Probability of Yes given Actual Yes: ", mean_yes_given_yes, "\n")


# ROC curve
roc_curve <- roc(test_data$wants_extra_baggage, test_prob[, "Yes"])
plot(roc_curve, print.thres="best", col = "blue", lwd = 2, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")


# AUC
auc(roc_curve)

### wants_preferred_seat ###
############################

### Fit the initial logistic regression model ###

# Initial logistic regression model
logist_model2 <- glm(wants_preferred_seat ~ . - wants_preferred_seat - wants_in_flight_meals, data = train_data_seat, family = "binomial")

# Stepwise backward elimination based on AIC
reduced_model2 <- stepAIC(logist_model2, direction = "backward")
summary(reduced_model2)

# Capture the formula of the reduced model
reduced_formula2 <- formula(reduced_model2)

# Convert 'wants_preferred_seat' from numeric to factor on the training set
train_data$wants_preferred_seat <- factor(train_data$wants_preferred_seat, levels = c(0, 1), labels = c("No", "Yes"))

# Convert 'wants_preferred_seat' from numeric to factor on the test set
test_data$wants_preferred_seat <- factor(test_data$wants_preferred_seat, levels = c(0, 1), labels = c("No", "Yes"))

### Cross-validation ###

# Train the model using cross-validation
cv_model2 <- train(
  reduced_formula2,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Model summary
summary(cv_model2)
print(cv_model2)

# Predict on the test data
test_prob2 <- predict(cv_model2, newdata = test_data, type = "prob")
test_pred2 <- ifelse(test_prob2[, "Yes"] > 0.357, "Yes", "No")

test_pred2 <- as.factor(test_pred2)
test_data$wants_preferred_seat <- as.factor(test_data$wants_preferred_seat)

# Confusion matrix
confusionMatrix(data=test_pred2, reference = test_data$wants_preferred_seat)

# Boxplot of the probabilities
boxplot(test_prob2[, "Yes"]~test_data$wants_preferred_seat, col = c("red", "blue"), xlab = "Wants Preferred Seat", ylab = "Probability of Yes", main = "Boxplot of Predicted Probabilities")

# Average predicted probability for those who did NOT want preferred seat
mean_yes_given_no2 <- mean(test_prob2[test_data$wants_preferred_seat == "No", "Yes"])

# Average predicted probability for those who DID want preferred seat
mean_yes_given_yes2 <- mean(test_prob2[test_data$wants_preferred_seat == "Yes", "Yes"])

# Output the averages
cat("Average Probability of Yes given Actual No: ", mean_yes_given_no2, "\n")
cat("Average Probability of Yes given Actual Yes: ", mean_yes_given_yes2, "\n")

# ROC curve
roc_curve2 <- roc(test_data$wants_preferred_seat, test_prob2[, "Yes"])
plot(roc_curve2, print.thres="best", col = "blue", lwd = 2, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

# AUC
auc(roc_curve2)

### wants_in_flight_meals ###
#############################

### Fit the initial logistic regression model ###

# Initial logistic regression model
logist_model3 <- glm(wants_in_flight_meals ~ ., data = train_data, family = "binomial")

# Stepwise backward elimination based on AIC
reduced_model3 <- stepAIC(logist_model3, direction = "backward")
summary(reduced_model3)

# Capture the formula of the reduced model
reduced_formula3 <- formula(reduced_model3)

# Convert 'wants_in_flight_meals' from numeric to factor on the training set
train_data$wants_in_flight_meals <- factor(train_data$wants_in_flight_meals, levels = c(0, 1), labels = c("No", "Yes"))

# Convert 'wants_in_flight_meals' from numeric to factor on the test set
test_data$wants_in_flight_meals <- factor(test_data$wants_in_flight_meals, levels = c(0, 1), labels = c("No", "Yes"))

### Cross-validation ###

# Train the model using cross-validation
cv_model3 <- train(
  reduced_formula3,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Model summary
summary(cv_model3)
print(cv_model3)

# Predict on the test data
test_prob3 <- predict(cv_model3, newdata = test_data, type = "prob")
test_pred3 <- ifelse(test_prob3[, "Yes"] > 0.443, "Yes", "No")

test_pred3 <- as.factor(test_pred3)
test_data$wants_in_flight_meals <- as.factor(test_data$wants_in_flight_meals)

# Confusion matrix
confusionMatrix(data=test_pred3, reference = test_data$wants_in_flight_meals)

# Boxplot of the probabilities
boxplot(test_prob3[, "Yes"]~test_data$wants_in_flight_meals, col = c("red", "blue"), xlab = "Wants In-Flight Meals", ylab = "Probability of Yes", main = "Boxplot of Predicted Probabilities")

# Average predicted probability for those who did NOT want in-flight meals
mean_yes_given_no3 <- mean(test_prob3[test_data$wants_in_flight_meals == "No", "Yes"])

# Average predicted probability for those who DID want in-flight meals
mean_yes_given_yes3 <- mean(test_prob3[test_data$wants_in_flight_meals == "Yes", "Yes"])

# Output the averages
cat("Average Probability of Yes given Actual No: ", mean_yes_given_no3, "\n")
cat("Average Probability of Yes given Actual Yes: ", mean_yes_given_yes3, "\n")

# ROC curve
roc_curve3 <- roc(test_data$wants_in_flight_meals, test_prob3[, "Yes"])
plot(roc_curve3, print.thres="best", col = "blue", lwd = 2, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

# AUC
auc(roc_curve3)