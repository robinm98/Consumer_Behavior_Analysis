#### MLBA Project - Random Forest ####
######################################

# Load the required libraries
library(rpart)
library(caret)
library(randomForest)
library(here)
library(tidyverse)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))

# Remove the columns that are not needed
data <- data |>
  select(-c("route", "departure", "arrival", "booking_origin"))

# Convert categorical variables to factors
categorical_vars <- c("sales_channel", "trip_type", "flight_day")
data <- data |>
  mutate(across(all_of(categorical_vars), as.factor))

# Combine the labels into a single feature
data$combined_label <- with(data, paste0(wants_extra_baggage, wants_in_flight_meals, wants_preferred_seat))

# Convert combined_label to a factor
data$combined_label <- as.factor(data$combined_label)

# Prepare the training and testing data
set.seed(123)
train_index <- createDataPartition(data$combined_label, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the Random Forest model on the combined label
rf_model <- randomForest(combined_label ~ . - wants_extra_baggage - wants_in_flight_meals - wants_preferred_seat, data = train_data, ntree = 100)

# Predict using the model
predictions <- predict(rf_model, test_data)

# Evaluate the model
confusion_matrix <- table(Predicted = predictions, Actual = test_data$combined_label)
print(confusion_matrix)

# confusion matrix to see how well the model performed
print(confusionMatrix(confusion_matrix))

##################################
#### Random Forest Multilabel ####
##################################

# Load the required libraries
library(caret)
library(randomForest)
library(here)
library(tidyverse)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))

# Remove the columns that are not needed
data <- data |>
  select(-c("route", "departure", "arrival", "booking_origin"))

# Convert categorical variables to factors
categorical_vars <- c("sales_channel", "trip_type", "flight_day")
data <- data |>
  mutate(across(all_of(categorical_vars), as.factor))

# Prepare the training and testing data
set.seed(123)
train_indices <- sample(seq_len(nrow(data)), size = floor(0.8 * nrow(data)))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# List of outcome variables
outcome_vars <- c("wants_extra_baggage", "wants_in_flight_meals", "wants_preferred_seat")

# Train a Random Forest model for each label
models <- lapply(outcome_vars, function(label) {
  formula <- as.formula(paste(label, "~ . - wants_extra_baggage - wants_in_flight_meals - wants_preferred_seat"))
  rf_model <- randomForest(formula, data = train_data, ntree = 100)
  return(rf_model)
})

# Predict using the models
predictions <- lapply(models, function(model, test_data) {
  predict(model, test_data)
}, test_data)

# Combine predictions into a single data frame
predictions_df <- do.call(cbind, predictions)
colnames(predictions_df) <- outcome_vars

# Evaluate the model (example using simple accuracy here, you can also use other metrics like F1 score, etc.)
results <- lapply(seq_along(models), function(i) {
  actual <- test_data[[outcome_vars[i]]]
  predicted <- predictions_df[, i]
  confusionMatrix <- confusionMatrix(factor(predicted, levels = levels(actual)), actual)
  return(confusionMatrix)
})

# Print results
print(results)