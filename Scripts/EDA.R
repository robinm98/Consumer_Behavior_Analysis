#### MLBA Project - EDA ####
############################

# Load the required libraries
library(readr)
library(here)
library(ggplot2)
library(ggthemes)
library(gridExtra)

# Load the data
data <- read.csv(here("Data", "cleaned_customer_booking.csv"))


### DATA DISTRIBUTION OF CATEGORICAL VALUES ###
###############################################

# Plotting the distribution of 'sales_channel'
ggplot(data, aes(x = sales_channel)) +
  geom_bar(fill = "blue") +
  theme_minimal() +
  labs(title = "Distribution of Sales Channel", x = "Sales Channel", y = "Count")

# Plotting the distribution of 'trip_type'
ggplot(data, aes(x = trip_type)) +
  geom_bar(fill = "blue") +
  theme_minimal() +
  labs(title = "Distribution of Trip Type", x = "Trip Type", y = "Count")

# Plotting the distribution of 'flight_day'
ggplot(data, aes(x = flight_day)) +
  geom_bar(fill = "blue") +
  theme_minimal() +
  labs(title = "Distribution of Flight Day", x = "Flight Day", y = "Count")

# Plotting the distribution of 'continent'
ggplot(data, aes(x = continent)) +
  geom_bar(fill = "blue") +
  theme_minimal() +
  labs(title = "Distribution of Booking Origin", x = "Booking Origin", y = "Count")

# Plotting the distribution of the top 10 departure airports
ggplot(data %>% 
         count(departure) %>%    # Count occurrences of each departure airport
         top_n(10, n),           # Select the top 10 airports by count
       aes(x = reorder(departure, n), y = n)) +  # Reorder factor levels by count for plotting
  geom_bar(stat = "identity", fill = "blue") +   # Use identity stat for pre-counted data
  theme_minimal() +
  labs(title = "Distribution of Top 10 Departure Airports",
       x = "Departure Airport", y = "Count") +
  scale_x_discrete(limits = function(x) x)  # Ensure the x-axis respects the order in the data

# Plotting the distribution of the top 10 arrival airports
ggplot(data %>% 
         count(arrival) %>%    # Count occurrences of each departure airport
         top_n(10, n),           # Select the top 10 airports by count
       aes(x = reorder(arrival, n), y = n)) +  # Reorder factor levels by count for plotting
  geom_bar(stat = "identity", fill = "blue") +   # Use identity stat for pre-counted data
  theme_minimal() +
  labs(title = "Distribution of Top 10 Departure Airports",
       x = "Departure Airport", y = "Count") +
  scale_x_discrete(limits = function(x) x)  # Ensure the x-axis respects the order in the data

# Plotting the distribution of 'booking_complete'
ggplot(data, aes(x = booking_complete)) +
  geom_bar(fill = "blue") +
  theme_minimal() +
  labs(title = "Distribution of Booking Completion", x = "Booking Completion", y = "Count")

# Plotting the distribution of 'wants_extra_baggage'
ggplot(data, aes(x = wants_extra_baggage)) +
  geom_bar(fill = "blue") +
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference", x = "Extra Baggage Preference", y = "Count")

# Plotting the distribution of 'wants_in_flight_meals'
ggplot(data, aes(x = wants_in_flight_meals)) +
  geom_bar(fill = "blue") +
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference", x = "In-Flight Meals Preference", y = "Count")

# Plotting the distribution of 'wants_preferred_seats'
ggplot(data, aes(x = wants_preferred_seat)) +
  geom_bar(fill = "blue") +
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference", x = "Preferred Seats Preference", y = "Count")

### DATA DISTRIBUTION OF NUMERICAL VALUES ###

# Plotting the distribution of 'purchase_lead', filtering within the plot
ggplot(data, aes(x = purchase_lead)) +
  geom_histogram(data = data[data$purchase_lead < 450, ], fill = "blue", bins = 50) +  # Apply filtering directly in geom_histogram
  scale_x_continuous(breaks = seq(from = 0, to = 500, by = 50), limits = c(0, 450)) +  # Adjust breaks and set limits to 500
  theme_minimal() +
  labs(title = "Distribution of Purchase Lead Time (up to 500 days)", 
       x = "Purchase Lead Time (days)", 
       y = "Count")


