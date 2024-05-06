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

### 3 models : Logistic Regression, Decision Tree, Random Forest
### 3 variable of interest : wants_extra_baggage, wants_in_flight_meal, wants_preferred_seat

# Convert categorical variables to factor
categorical_vars <- c("sales_channel", "trip_type", "flight_day", "continent")
data <- data |> 
  mutate(across(all_of(categorical_vars), as.factor))

# Prepare dummy variables
data <- data |> 
  dummy_cols(select_columns = categorical_vars, remove_first_dummy = TRUE)

# Remove unneeded columns if they are not used in any models (verify first)
testdata <- data |> 
  select(-route, -booking_origin, -departure, -arrival, -flight_day, -continent, -sales_channel, -trip_type)

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

# Setup cross-validation on training data
train_control <- trainControl(
  method = "cv",
  number = 10,  # number of folds in cross-validation
  savePredictions = "final",
  classProbs = TRUE  # since it's a classification problem
)

# Train the model using cross-validation
cv_model <- train(
  wants_extra_baggage ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Print the results
print(cv_model)

# Predict on test set
predictions <- predict(cv_model, newdata = test_data, type = "raw")
confusionMatrix(data = predictions, reference = test_data$wants_extra_baggage)

# Calculate AUC
probabilities <- predict(cv_model, newdata = test_data, type = "prob")
auc <- AUC(probabilities, test_data$wants_extra_baggage)
print(auc)