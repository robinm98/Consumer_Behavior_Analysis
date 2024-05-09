#### MLBA Project - NN ####
############################

# Load the required libraries
library(reticulate)
library(tidyverse)
library(keras)
library(here)
library(caret)

# Create virtual environment for Python
virtualenv_create(envname = "r-tensorflow", python = "auto")
use_virtualenv("r-tensorflow", required = TRUE)

# Install necessary Python packages within the environment
py_install(c("numpy", "pandas", "tensorflow", "keras"), envname = "r-tensorflow")

# Initialize Python modules
k <- import("keras")
np <- import("numpy")

############################
#### Multilabel ############
############################

# Read the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))

# Prepare data by removing unnecessary columns and converting categorical variables
data <- data |>
  select(-c("route", "departure", "arrival", "booking_origin")) |>  # Keep label columns
  mutate(across(c("sales_channel", "trip_type", "flight_day"), as.factor)) |>
  mutate(across(c("sales_channel", "trip_type", "flight_day"), as.integer))  # Convert factors to integers for embedding or input

# Ensure labels are binary
data$wants_extra_baggage <- as.integer(data$wants_extra_baggage)
data$wants_in_flight_meals <- as.integer(data$wants_in_flight_meals)
data$wants_preferred_seat <- as.integer(data$wants_preferred_seat)

set.seed(123)  # For reproducibility
train_indices <- createDataPartition(y = data$wants_extra_baggage, p = 0.8, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Prepare a preProcess scaler
preProcValues <- preProcess(train_data[, -c(ncol(train_data)-2:ncol(train_data))], method = c("center", "scale"))

# Apply the scaler
train_data[, -c(ncol(train_data)-2:ncol(train_data))] <- predict(preProcValues, train_data[, -c(ncol(train_data)-2:ncol(train_data))])
test_data[, -c(ncol(train_data)-2:ncol(train_data))] <- predict(preProcValues, test_data[, -c(ncol(train_data)-2:ncol(train_data))])

model <- keras_model_sequential() |>
  layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(train_data)-3)) |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 64, activation = 'relu') |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 3, activation = 'sigmoid')  # 3 outputs corresponding to 3 labels

# Compile the model
model |> compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)

# Extract labels into a matrix
train_labels <- as.matrix(train_data[, c("wants_extra_baggage", "wants_in_flight_meals", "wants_preferred_seat")])
test_labels <- as.matrix(test_data[, c("wants_extra_baggage", "wants_in_flight_meals", "wants_preferred_seat")])

# Ensure features and labels are correctly aligned
train_features <- as.matrix(train_data[, -c(ncol(train_data)-2:ncol(train_data))])
test_features <- as.matrix(test_data[, -c(ncol(test_data)-2:ncol(test_data))])

# Fit the model
history <- model |> fit(
  train_features,
  train_labels,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model
model |> evaluate(test_features, test_labels)

# Make predictions
predictions <- model |> predict(test_features)

# Threshold the predictions to convert probabilities to binary decisions
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
colnames(predicted_classes) <- c("wants_extra_baggage", "wants_in_flight_meals", "wants_preferred_seat")

# View some of the predictions
head(predicted_classes)