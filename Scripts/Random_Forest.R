#### MLBA Project - Random Forest ####
######################################

library(randomForest)
library(caret)
library(here)
library(dplyr)
library(tidyr)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))

# Remove the columns that are not needed and convert categorical variables
data <- data %>%
  select(-c("route", "departure", "arrival", "booking_origin")) %>%
  mutate(across(c("sales_channel", "trip_type", "flight_day"), as.factor))

# Combine the labels into a single feature and convert to a factor with valid names
data$combined_label <- factor(paste0(data$wants_extra_baggage, data$wants_in_flight_meals, data$wants_preferred_seat))
levels(data$combined_label) <- make.names(levels(data$combined_label))

# Identify and upsample minority classes
minority_classes <- c("001", "010", "011", "101")
upsampled_data <- data %>%
  group_by(combined_label) %>%
  mutate(n = if_else(combined_label %in% minority_classes, 5, 1)) %>%
  ungroup() %>%
  slice(rep(row_number(), n)) %>%
  select(-n)

# Prepare the training and testing data
set.seed(123)
train_index <- createDataPartition(upsampled_data$combined_label, p = 0.8, list = FALSE)
train_data <- upsampled_data[train_index, ]
test_data <- data[-train_index, ]

# Define the control function using cross-validation
fit_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE
)

# Define the tuning grid without num.trees
tuning_grid <- expand.grid(
  mtry = c(sqrt(ncol(train_data) - 1), ncol(train_data)/3),
  splitrule = "gini",
  min.node.size = c(1, 5, 10)
)

# Train the model with a specified number of trees
rf_model_tuned <- train(
  combined_label ~ . - wants_extra_baggage - wants_in_flight_meals - wants_preferred_seat,
  data = train_data,
  method = "ranger",
  trControl = fit_control,
  tuneGrid = tuning_grid,
  metric = "Accuracy",
  num.trees = 500
)

# Predict using the model
predictions <- predict(rf_model_tuned, newdata = test_data)

# Evaluate the model
confusion_matrix <- table(Predicted = predictions, Actual = test_data$combined_label)
print(confusion_matrix)
print(confusionMatrix(confusion_matrix))