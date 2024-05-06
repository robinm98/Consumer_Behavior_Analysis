#### MLBA Project - Modeling ####
#################################

# Load the required libraries
library(rpart)
library(caret)
library(randomForest)
library(here)
library(tidyverse)
library(ModelMetrics)
library(fastDummies)
library(MASS)
library(dplyr)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))

# show unique values in route column
unique(data$route)

### 3 models : Logistic Regression, Decision Tree, Random Forest
### 3 variable of interest : wants_extra_baggage, wants_in_flight_meal, wants_preferred_seat

#################################
# Logistic Regression ###########
#################################

# Convert categorical variables to factor
categorical_vars <- c("sales_channel", "trip_type", "flight_day", "continent")
data <- data |> 
  mutate(across(all_of(categorical_vars), as.factor))

# Prepare dummy variables
data <- data |> 
  dummy_cols(select_columns = categorical_vars, remove_first_dummy = TRUE)

# Remove unneeded columns if they are not used in any models (verify first), do we want to keep wants preffered seat and meal?
testdata <- data |> 
  dplyr::select(-route, -booking_origin, -departure, -arrival, -flight_day, -continent, -sales_channel, -trip_type) 
                
# Ensure the target variable is a factor
testdata$wants_extra_baggage <- as.factor(testdata$wants_extra_baggage)

# Splitting data into training and testing sets
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(testdata$wants_extra_baggage, p = 0.8, list = FALSE)
train_data <- testdata[trainIndex, ]
test_data <- testdata[-trainIndex, ]

# Initial logistic regression model
logist_model1 <- glm(wants_extra_baggage ~ ., data = train_data, family = "binomial")

# Stepwise backward elimination based on AIC
reduced_model <- stepAIC(logist_model1, direction = "backward")
summary(reduced_model)

# Capture the formula of the reduced model
reduced_formula <- formula(reduced_model)

# Adjust the factor levels of the target variable
train_data$wants_extra_baggage <- factor(train_data$wants_extra_baggage, levels = c(0, 1), labels = c("Class0", "Class1"))

# Make sure to apply the same change to the test_data if you are going to use it later
test_data$wants_extra_baggage <- factor(test_data$wants_extra_baggage, levels = c(0, 1), labels = c("Class0", "Class1"))

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

# Print the results
print(cv_model)

#################################
# Decision Tree #################
#################################

# Train the decision tree model
tree_model <- rpart(wants_extra_baggage ~ ., data = train_data, method = "class")

# View the tree structure
print(tree_model)

# Train a decision tree with a maximum depth to prevent overfitting
tree_model_tuned <- rpart(wants_extra_baggage ~ ., data = train_data, method = "class",
                          control = rpart.control(maxdepth = 5))

# Evaluate the model using a confusion matrix
caret::confusionMatrix(tree_predictions, test_data$wants_extra_baggage)

# Visualize the tuned tree
plot(tree_model_tuned)
text(tree_model_tuned, use.n = TRUE)

# Setup for cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the model using cross-validation
cv_tree_model <- train(wants_extra_baggage ~ ., data = train_data, method = "rpart",
                       trControl = train_control,
                       tuneLength = 10)  # Tune the complexity parameter

# Print the results
print(cv_tree_model)

#################################
# Random Forest #################
#################################

# Train the Random Forest model
rf_model <- randomForest(wants_extra_baggage ~ ., data = train_data, ntree = 500, mtry = floor(sqrt(ncol(train_data))), importance = TRUE)

# Print the model summary
print(rf_model)

# Predict using the Random Forest model
rf_predictions <- predict(rf_model, test_data)

# Evaluate the model using a confusion matrix
caret::confusionMatrix(rf_predictions, test_data$wants_extra_baggage)

# Check variable importance
varImpPlot(rf_model)

# Setup for cross-validation
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE, savePredictions = "final")

# Train the model using cross-validation with tuning
rf_tuned_model <- train(wants_extra_baggage ~ ., data = train_data, method = "rf",
                        trControl = train_control,
                        tuneGrid = expand.grid(.mtry = c(2, floor(sqrt(ncol(train_data))), ncol(train_data)/2)),
                        ntree = 500)

# Print the results of the tuned model
print(rf_tuned_model)

#################################
# Metrics ######################
#################################
# Predict on test set
predictions <- predict(cv_model, newdata = test_data, type = "raw")
caret::confusionMatrix(data = predictions, reference = test_data$wants_extra_baggage)

# Assuming `probabilities` has two columns for the two classes 'No' and 'Yes'
# Here, let's say the second column of `probabilities` corresponds to 'Yes'
test_data$predicted_prob_yes = probabilities[, 2]
test_data$predicted_class = ifelse(test_data$predicted_prob_yes >= 0.5, "Yes", "No")


df_pred_lr <- data.frame(
  obs = test_data$wants_extra_baggage,   # Observed classes
  Yes = test_data$predicted_prob_yes,    # Predicted probabilities for 'Yes'
  No = 1 - test_data$predicted_prob_yes, # Predicted probabilities for 'No'
  pred = as.factor(test_data$predicted_class)  # Predicted class labels
)

# You need a dummy trainControl object for twoClassSummary, as it's typically used inside train functions
dummy_control <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)

# Assuming the classes are "Yes" and "No"
# First, ensure both are factors and have the same levels
df_pred_lr$obs <- factor(df_pred_lr$obs, levels = c("No", "Yes"))
df_pred_lr$pred <- factor(df_pred_lr$pred, levels = c("No", "Yes"))

# Now apply twoClassSummary
summary_stats <- twoClassSummary(df_pred_lr, lev = levels(df_pred_lr$obs))
print(summary_stats)

# Make sure to load pROC
library(pROC)

# Generating ROC curve
ROC_curve <- roc(response = df_pred_lr$obs, predictor = df_pred_lr$Yes)

# Plotting ROC curve with the best threshold marked
plot(ROC_curve, print.thres = "best")