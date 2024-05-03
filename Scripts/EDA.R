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

### BAR CHART ###

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
  scale_x_continuous(breaks = c(0, 1)) +
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

### PROPORTION OF INTEREST VARIABLES ###
########################################

# Plot the proportion of 'wants_extra_baggage' with a Pie Chart
# Calculate proportions
prop_table <- table(data$wants_extra_baggage)
prop <- prop.table(prop_table)

# Convert proportions to percentage
prop_percent <- prop * 100
prop_labels <- paste0(prop_percent, "%")

# Plot pie chart
pie(prop, 
    main = "Proportion of Extra Baggage Preference",
    labels = prop_labels,
    col = c("red", "blue"),
    clockwise = TRUE
)

# Add legend
legend("bottom", legend = c("No", "Yes"), fill = c("red", "blue"), title = "Extra Baggage Preference", horiz = TRUE)

# Plot the proportion of 'wants_in_flight_meals' with a Pie Chart
# Calculate proportions
prop_table <- table(data$wants_in_flight_meals)
prop <- prop.table(prop_table)

# Convert proportions to percentage
prop_percent <- prop * 100
prop_labels <- paste0(prop_percent, "%")

# Plot pie chart
pie(prop, 
    main = "Proportion of In-Flight Meals Preference",
    labels = prop_labels,
    col = c("red", "blue"),
    clockwise = TRUE
)

# Add legend
legend("bottom", legend = c("No", "Yes"), fill = c("red", "blue"), title = "In-Flight Meals Preference", horiz = TRUE)


# Plot the proportion of 'wants_preferred_seats' with a Pie Chart
# Calculate proportions
prop_table <- table(data$wants_preferred_seat)
prop <- prop.table(prop_table)

# Convert proportions to percentage
prop_percent <- prop * 100
prop_labels <- paste0(prop_percent, "%")

# Plot pie chart
pie(prop, 
    main = "Proportion of Preferred Seats Preference",
    labels = prop_labels,
    col = c("red", "blue"),
    clockwise = TRUE
)

# Add legend
legend("bottom", legend = c("No", "Yes"), fill = c("red", "blue"), title = "Preferred Seats Preference", horiz = TRUE)


### DATA DISTRIBUTION OF NUMERICAL VALUES ###
#############################################

### HISTOGRAM ###

# Plotting the distribution of 'purchase_lead'
ggplot(data, aes(x = purchase_lead)) +
  geom_histogram(data = data[data$purchase_lead < 450, ], fill = "blue", bins = 50) +  # Apply filtering directly in geom_histogram
  scale_x_continuous(breaks = seq(from = 0, to = 500, by = 50), limits = c(0, 450)) +  # Adjust breaks and set limits to 500
  theme_minimal() +
  labs(title = "Distribution of Purchase Lead Time (up to 500 days)", 
       x = "Purchase Lead Time (days)", 
       y = "Count")

# Plotting the distribution of 'flight_duration'
ggplot(data, aes(x = flight_duration)) +
  geom_histogram(fill = "blue", bins =50) +  
  scale_x_continuous(breaks = seq(0, max(data$flight_duration), by = 0.50)) +
  theme_minimal() +
  labs(title = "Distribution of Flight Duration", 
       x = "Flight Duration (hours)", 
       y = "Count")

# Plotting the distribution of 'num_passengers'
ggplot(data, aes(x = as.factor(num_passengers))) +
  geom_bar(fill = "blue") +  
  scale_x_discrete(labels = c("1", "2", "3", "4", "5", "6", "7", "8", "9")) +
  theme_minimal() +
  labs(title = "Distribution of Number of Passengers", 
       x = "Number of Passengers", 
       y = "Count")

# Plotting the distribution of 'flight_hour'
ggplot(data, aes(x = flight_hour)) +
  geom_bar(fill = "blue") +  
  scale_x_continuous(breaks = seq(0, 24, by = 1)) +
  theme_minimal() +
  labs(title = "Distribution of Flight Hour", 
       x = "Flight Hour", 
       y = "Count")

# Plot the distribution of 'length_of_stay'
ggplot(data[data$length_of_stay < 250,], aes(x = length_of_stay)) +
  geom_histogram(fill = "blue", bins = 25) +
  scale_x_continuous(breaks = seq(0, 250, by = 25)) +
  theme_minimal() +
  labs(title = "Distribution of Length of Stay", x = "Length of Stay (days)", y = "Count")


### DATA DISTRIBUTION OF INTEREST VARIABLES BY CATEGORICAL VARIABLES ###
########################################################################

### BAR CHART ###

### SALES CHANNEL

# Plotting the distribution of 'wants_extra_baggage' by 'sales_channel'
ggplot(data, aes(x = wants_extra_baggage)) +
  geom_bar(fill = "blue") +
  facet_wrap(~sales_channel) +  # Facet by 'sales_channel'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Sales Channel", 
       x = "Extra Baggage Preference", 
       y = "Count")


# Plotting the distribution of 'wants_in_flight_meals' by 'sales_channel'
ggplot(data, aes(x = wants_in_flight_meals)) +
  geom_bar(fill = "blue") +
  facet_wrap(~sales_channel) +  # Facet by 'sales_channel'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Sales Channel", 
       x = "In-Flight Meals Preference", 
       y = "Count")

# Plotting the distribution of 'wants_preferred_seats' by 'sales_channel'
ggplot(data, aes(x = wants_preferred_seat)) +
  geom_bar(fill = "blue") +
  facet_wrap(~sales_channel) +  # Facet by 'sales_channel'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Sales Channel", 
       x = "Preferred Seats Preference", 
       y = "Count")

### TRIP TYPE

# Plotting the distribution of 'wants_extra_baggage' by 'trip_type'
ggplot(data, aes(x = wants_extra_baggage)) +
  geom_bar(fill = "blue") +
  facet_wrap(~trip_type) +  # Facet by 'trip_type'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Trip Type", 
       x = "Extra Baggage Preference", 
       y = "Count")

# Plotting the distribution of 'wants_in_flight_meals' by 'trip_type'
ggplot(data, aes(x = wants_in_flight_meals)) +
  geom_bar(fill = "blue") +
  facet_wrap(~trip_type) +  # Facet by 'trip_type'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Trip Type", 
       x = "In-Flight Meals Preference", 
       y = "Count")

# Plotting the distribution of 'wants_preferred_seats' by 'trip_type'
ggplot(data, aes(x = wants_preferred_seat)) +
  geom_bar(fill = "blue") +
  facet_wrap(~trip_type) +  # Facet by 'trip_type'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Trip Type", 
       x = "Preferred Seats Preference", 
       y = "Count")

### CONTINENT

# Plotting the distribution of 'wants_extra_baggage' by 'continent'
ggplot(data, aes(x = wants_extra_baggage)) +
  geom_bar(fill = "blue") +
  facet_wrap(~continent) +  # Facet by 'continent'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Continent", 
       x = "Extra Baggage Preference", 
       y = "Count")

# Plotting the distribution of 'wants_in_flight_meals' by 'continent'
ggplot(data, aes(x = wants_in_flight_meals)) +
  geom_bar(fill = "blue") +
  facet_wrap(~continent) +  # Facet by 'continent'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Continent", 
       x = "In-Flight Meals Preference", 
       y = "Count")

# Plotting the distribution of 'wants_preferred_seats' by 'continent'
ggplot(data, aes(x = wants_preferred_seat)) +
  geom_bar(fill = "blue") +
  facet_wrap(~continent) +  # Facet by 'continent'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Continent", 
       x = "Preferred Seats Preference", 
       y = "Count")

### BOOKING COMPLETE

# Plotting the distribution of 'wants_extra_baggage' by 'booking_complete'
ggplot(data, aes(x = wants_extra_baggage)) +
  geom_bar(fill = "blue") +
  facet_wrap(~booking_complete) +  # Facet by 'booking_complete'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Booking Completion", 
       x = "Extra Baggage Preference", 
       y = "Count")

# Plotting the distribution of 'wants_in_flight_meals' by 'booking_complete'
ggplot(data, aes(x = wants_in_flight_meals)) +
  geom_bar(fill = "blue") +
  facet_wrap(~booking_complete) +  # Facet by 'booking_complete'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Booking Completion", 
       x = "In-Flight Meals Preference", 
       y = "Count")

# Plotting the distribution of 'wants_preferred_seats' by 'booking_complete'
ggplot(data, aes(x = wants_preferred_seat)) +
  geom_bar(fill = "blue") +
  facet_wrap(~booking_complete) +  # Facet by 'booking_complete'
  scale_x_continuous(breaks = c(0, 1)) + 
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Booking Completion", 
       x = "Preferred Seats Preference", 
       y = "Count")

### DATA DISTRIBUTION OF INTEREST VARIABLES BY NUMERICAL VARIABLES ###
######################################################################

### BOX PLOT ###

### PURCHASE LEAD TIME

# Plotting the distribution of 'wants_extra_baggage' by 'purchase_lead' with boxplot
ggplot(data[data$purchase_lead < 250,], aes(x = as.factor(wants_extra_baggage), y = purchase_lead)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$purchase_lead), by = 50)) +
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Purchase Lead Time", 
       x = "Extra Baggage Preference", 
       y = "Purchase Lead Time (days)")


# Plotting the distribution of 'wants_in_flight_meals' by 'purchase_lead' with boxplot
ggplot(data[data$purchase_lead < 250,], aes(x = as.factor(wants_in_flight_meals), y = purchase_lead)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$purchase_lead), by = 50)) +
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Purchase Lead Time", 
       x = "In-Flight Meals Preference", 
       y = "Purchase Lead Time (days)")

# Plotting the distribution of 'wants_preferred_seats' by 'purchase_lead' with boxplot
ggplot(data[data$purchase_lead < 250,], aes(x = as.factor(wants_preferred_seat), y = purchase_lead)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$purchase_lead), by = 50)) +
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Purchase Lead Time", 
       x = "Preferred Seats Preference", 
       y = "Purchase Lead Time (days)")

### LENGTH OF STAY

# Plotting the distribution of 'wants_extra_baggage' by 'lenght_of_stay' with boxplot
ggplot(data[data$length_of_stay < 75,], aes(x = as.factor(wants_extra_baggage), y = length_of_stay)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$length_of_stay), by = 25)) +
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Length of Stay", 
       x = "Extra Baggage Preference", 
       y = "Length of Stay (days)")

# Plotting the distribution of 'wants_in_flight_meals' by 'lenght_of_stay' with boxplot
ggplot(data[data$length_of_stay < 75,], aes(x = as.factor(wants_in_flight_meals), y = length_of_stay)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$length_of_stay), by = 25)) +
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Length of Stay", 
       x = "In-Flight Meals Preference", 
       y = "Length of Stay (days)")

# Plotting the distribution of 'wants_preferred_seats' by 'lenght_of_stay' with boxplot
ggplot(data[data$length_of_stay < 75,], aes(x = as.factor(wants_preferred_seat), y = length_of_stay)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  scale_y_continuous(breaks = seq(0, max(data$length_of_stay), by = 25)) +
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Length of Stay", 
       x = "Preferred Seats Preference", 
       y = "Length of Stay (days)")

### FLIGHT DURATION

# Plotting the distribution of 'wants_extra_baggage' by 'flight_duration' with boxplot
ggplot(data, aes(x = as.factor(wants_extra_baggage), y = flight_duration)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  theme_minimal() +
  labs(title = "Distribution of Extra Baggage Preference by Flight Duration", 
       x = "Extra Baggage Preference", 
       y = "Flight Duration (hours)")

# Plotting the distribution of 'wants_in_flight_meals' by 'flight_duration' with boxplot
ggplot(data, aes(x = as.factor(wants_in_flight_meals), y = flight_duration)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  theme_minimal() +
  labs(title = "Distribution of In-Flight Meals Preference by Flight Duration", 
       x = "In-Flight Meals Preference", 
       y = "Flight Duration (hours)")

# Plotting the distribution of 'wants_preferred_seats' by 'flight_duration' with boxplot
ggplot(data, aes(x = as.factor(wants_preferred_seat), y = flight_duration)) +
  geom_boxplot(fill = "blue") +
  scale_x_discrete(labels = c("No", "Yes")) +
  theme_minimal() +
  labs(title = "Distribution of Preferred Seats Preference by Flight Duration", 
       x = "Preferred Seats Preference", 
       y = "Flight Duration (hours)")

### PROPORTIONNAL BAR CHART ###

### NUMBER OF PASSENGERS

# Plotting the distribution of 'wants_extra_baggage' by 'num_passengers'
ggplot(data, aes(x = as.factor(num_passengers), fill = as.factor(wants_extra_baggage))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Extra Baggage") +
  labs(title = "Proportion of Extra Baggage Preference by Number of Passengers",
       x = "Number of Passengers",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_in_flight_meals' by 'num_passengers'
ggplot(data, aes(x = as.factor(num_passengers), fill = as.factor(wants_in_flight_meals))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "In-Flight Meals") +
  labs(title = "Proportion of In-Flight Meals Preference by Number of Passengers",
       x = "Number of Passengers",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_preferred_seats' by 'num_passengers'
ggplot(data, aes(x = as.factor(num_passengers), fill = as.factor(wants_preferred_seat))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Preferred Seats") +
  labs(title = "Proportion of Preferred Seats Preference by Number of Passengers",
       x = "Number of Passengers",
       y = "Proportion") +
  theme_minimal()

### FLIGHT DURATION

# Plotting the distribution of 'wants_extra_baggage' by 'flight_duration'
ggplot(data, aes(x = as.factor(flight_duration), fill = as.factor(wants_extra_baggage))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Extra Baggage") +
  labs(title = "Proportion of Extra Baggage Preference by Flight Duration",
       x = "Flight Duration (hours)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_in_flight_meals' by 'flight_duration'
ggplot(data, aes(x = as.factor(flight_duration), fill = as.factor(wants_in_flight_meals))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "In-Flight Meals") +
  labs(title = "Proportion of In-Flight Meals Preference by Flight Duration",
       x = "Flight Duration (hours)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_preferred_seats' by 'flight_duration'
ggplot(data, aes(x = as.factor(flight_duration), fill = as.factor(wants_preferred_seat))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Preferred Seats") +
  labs(title = "Proportion of Preferred Seats Preference by Flight Duration",
       x = "Flight Duration (hours)",
       y = "Proportion") +
  theme_minimal()

### LENGTH OF STAY

# Plotting the distribution of 'wants_extra_baggage' by 'length_of_stay'
ggplot(data, aes(x = as.factor(length_of_stay), fill = as.factor(wants_extra_baggage))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Extra Baggage") +
  labs(title = "Proportion of Extra Baggage Preference by Length of Stay",
       x = "Length of Stay (days)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_in_flight_meals' by 'length_of_stay'
ggplot(data, aes(x = as.factor(length_of_stay), fill = as.factor(wants_in_flight_meals))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "In-Flight Meals") +
  labs(title = "Proportion of In-Flight Meals Preference by Length of Stay",
       x = "Length of Stay (days)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_preferred_seats' by 'length_of_stay'
ggplot(data, aes(x = as.factor(length_of_stay), fill = as.factor(wants_preferred_seat))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Preferred Seats") +
  labs(title = "Proportion of Preferred Seats Preference by Length of Stay",
       x = "Length of Stay (days)",
       y = "Proportion") +
  theme_minimal()

### PURCHASE LEAD TIME

# Plotting the distribution of 'wants_extra_baggage' by 'purchase_lead'
ggplot(data, aes(x = as.factor(purchase_lead), fill = as.factor(wants_extra_baggage))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Extra Baggage") +
  labs(title = "Proportion of Extra Baggage Preference by Purchase Lead Time",
       x = "Purchase Lead Time (days)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_in_flight_meals' by 'purchase_lead'
ggplot(data, aes(x = as.factor(purchase_lead), fill = as.factor(wants_in_flight_meals))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "In-Flight Meals") +
  labs(title = "Proportion of In-Flight Meals Preference by Purchase Lead Time",
       x = "Purchase Lead Time (days)",
       y = "Proportion") +
  theme_minimal()

# Plotting the distribution of 'wants_preferred_seats' by 'purchase_lead'
ggplot(data, aes(x = as.factor(purchase_lead), fill = as.factor(wants_preferred_seat))) +
  geom_bar(position = "fill") +  # 'fill' stacks proportions to sum to 1
  scale_fill_manual(values = c("red", "blue"), labels = c("No", "Yes"), name = "Preferred Seats") +
  labs(title = "Proportion of Preferred Seats Preference by Purchase Lead Time",
       x = "Purchase Lead Time (days)",
       y = "Proportion") +
  theme_minimal()


### RELATIONSHIP BETWEEN NUMERICAL VARIABLES ###
################################################

### SCATTER PLOT ###

# Plotting the relationship between 'purchase_lead' and 'length_of_stay'
ggplot(data, aes(x = purchase_lead, y = length_of_stay)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  theme_minimal() +
  labs(title = "Relationship between Purchase Lead Time and Length of Stay", 
       x = "Purchase Lead Time (days)", 
       y = "Length of Stay (days)")


# Plotting the relationship between 'flight_duration' and 'length_of_stay'
ggplot(data, aes(x = flight_duration, y = length_of_stay)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  theme_minimal() +
  labs(title = "Relationship between Flight Duration and Length of Stay", 
       x = "Flight Duration (hours)", 
       y = "Length of Stay (days)")

# Plotting the relationship between 'flight_duration' and 'purchase_lead'
ggplot(data, aes(x = flight_duration, y = purchase_lead)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  theme_minimal() +
  labs(title = "Relationship between Flight Duration and Purchase Lead Time", 
       x = "Flight Duration (hours)", 
       y = "Purchase Lead Time (days)")

# Plotting the relationship between 'lenght_of_stay' and 'num_passengers'
ggplot(data, aes(x = length_of_stay, y = num_passengers)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  scale_y_continuous(limits = c(1, 9), breaks = seq(1, 9, by = 1)) +  # Set y-axis limits and breaks
  theme_minimal() +
  labs(title = "Relationship between Length of Stay and Number of Passengers", 
       x = "Length of Stay (days)", 
       y = "Number of Passengers")

# Plotting the relationship between 'flight_duration' and 'num_passengers'
ggplot(data, aes(x = flight_duration, y = num_passengers)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  scale_y_continuous(limits = c(1, 9), breaks = seq(1, 9, by = 1)) +  # Set y-axis limits and breaks
  theme_minimal() +
  labs(title = "Relationship between Flight Duration and Number of Passengers", 
       x = "Flight Duration (hours)", 
       y = "Number of Passengers")

# Plotting the relationship between 'purchase_lead' and 'num_passengers'
ggplot(data, aes(x = purchase_lead, y = num_passengers)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) + 
  scale_y_continuous(limits = c(1, 9), breaks = seq(1, 9, by = 1)) +  # Set y-axis limits and breaks
  theme_minimal() +
  labs(title = "Relationship between Purchase Lead Time and Number of Passengers", 
       x = "Purchase Lead Time (days)", 
       y = "Number of Passengers")


### RELATIONSHIP WITH VARIABLES OF INTEREST WITH NUMERICAL ###
##############################################################

### SCATTER PLOT ###

### NUMBER OF PASSENGERS

# Calculate proportion of 'Yes' for extra baggage per number of passengers
extra_baggage_rate <- data %>%
  group_by(num_passengers) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between number of passengers and extra baggage rates
ggplot(extra_baggage_rate, aes(x = num_passengers, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Number of Passengers and Extra Baggage Rates",
       x = "Number of Passengers",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per number of passengers
in_flight_meals_rate <- data %>%
  group_by(num_passengers) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between number of passengers and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = num_passengers, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Number of Passengers and In-Flight Meals Rates",
       x = "Number of Passengers",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per number of passengers
preferred_seats_rate <- data %>%
  group_by(num_passengers) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between number of passengers and preferred seats rates
ggplot(preferred_seats_rate, aes(x = num_passengers, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Number of Passengers and Preferred Seats Rates",
       x = "Number of Passengers",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### FLIGHT DURATION

# Calculate proportion of 'Yes' for extra baggage per flight duration
extra_baggage_rate <- data %>%
  group_by(flight_duration) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between flight duration and extra baggage rates
ggplot(extra_baggage_rate, aes(x = flight_duration, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Flight Duration and Extra Baggage Rates",
       x = "Flight Duration (hours)",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per flight duration
in_flight_meals_rate <- data %>%
  group_by(flight_duration) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between flight duration and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = flight_duration, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Flight Duration and In-Flight Meals Rates",
       x = "Flight Duration (hours)",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per flight duration
preferred_seats_rate <- data %>%
  group_by(flight_duration) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between flight duration and preferred seats rates
ggplot(preferred_seats_rate, aes(x = flight_duration, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Flight Duration and Preferred Seats Rates",
       x = "Flight Duration (hours)",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### PURCHASE LEAD TIME

# Calculate proportion of 'Yes' for extra baggage per purchase lead time
extra_baggage_rate <- data %>%
  group_by(purchase_lead) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between purchase lead time and extra baggage rates
ggplot(extra_baggage_rate, aes(x = purchase_lead, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Purchase Lead Time and Extra Baggage Rates",
       x = "Purchase Lead Time (days)",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per purchase lead time
in_flight_meals_rate <- data %>%
  group_by(purchase_lead) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between purchase lead time and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = purchase_lead, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Purchase Lead Time and In-Flight Meals Rates",
       x = "Purchase Lead Time (days)",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per purchase lead time
preferred_seats_rate <- data %>%
  group_by(purchase_lead) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between purchase lead time and preferred seats rates
ggplot(preferred_seats_rate, aes(x = purchase_lead, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Purchase Lead Time and Preferred Seats Rates",
       x = "Purchase Lead Time (days)",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### LENGTH OF STAY

# Calculate proportion of 'Yes' for extra baggage per length of stay
extra_baggage_rate <- data %>%
  group_by(length_of_stay) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between length of stay and extra baggage rates
ggplot(extra_baggage_rate, aes(x = length_of_stay, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Length of Stay and Extra Baggage Rates",
       x = "Length of Stay (days)",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per length of stay
in_flight_meals_rate <- data %>%
  group_by(length_of_stay) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between length of stay and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = length_of_stay, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Length of Stay and In-Flight Meals Rates",
       x = "Length of Stay (days)",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per length of stay
preferred_seats_rate <- data %>%
  group_by(length_of_stay) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between length of stay and preferred seats rates
ggplot(preferred_seats_rate, aes(x = length_of_stay, y = yes_rate)) +
  geom_point() +
  geom_smooth(method = "lm") +  # Adds a linear regression line
  labs(title = "Relationship Between Length of Stay and Preferred Seats Rates",
       x = "Length of Stay (days)",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### RELATIONSHIP WITH INTEREST VARIABLES WITH CATEGORICAL VARIABLES ###
#######################################################################

### BAR PLOT ###

### TRIP TYPE

# Calculate proportion of 'Yes' for extra baggage per trip type
extra_baggage_rate <- data %>%
  group_by(trip_type) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between trip type and extra baggage rates
ggplot(extra_baggage_rate, aes(x = trip_type, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Trip Type and Extra Baggage Rates",
       x = "Trip Type",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per trip type
in_flight_meals_rate <- data %>%
  group_by(trip_type) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between trip type and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = trip_type, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Trip Type and In-Flight Meals Rates",
       x = "Trip Type",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per trip type
preferred_seats_rate <- data %>%
  group_by(trip_type) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between trip type and preferred seats rates
ggplot(preferred_seats_rate, aes(x = trip_type, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Trip Type and Preferred Seats Rates",
       x = "Trip Type",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### CONTINENT

# Calculate proportion of 'Yes' for extra baggage per continent
extra_baggage_rate <- data %>%
  group_by(continent) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between continent and extra baggage rates
ggplot(extra_baggage_rate, aes(x = continent, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Continent and Extra Baggage Rates",
       x = "Continent",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per continent
in_flight_meals_rate <- data %>%
  group_by(continent) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between continent and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = continent, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Continent and In-Flight Meals Rates",
       x = "Continent",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per continent
preferred_seats_rate <- data %>%
  group_by(continent) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between continent and preferred seats rates
ggplot(preferred_seats_rate, aes(x = continent, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Continent and Preferred Seats Rates",
       x = "Continent",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### SALES CHANNEL

# Calculate proportion of 'Yes' for extra baggage per sales channel
extra_baggage_rate <- data %>%
  group_by(sales_channel) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between sales channel and extra baggage rates
ggplot(extra_baggage_rate, aes(x = sales_channel, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Sales Channel and Extra Baggage Rates",
       x = "Sales Channel",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per sales channel
in_flight_meals_rate <- data %>%
  group_by(sales_channel) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between sales channel and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = sales_channel, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Sales Channel and In-Flight Meals Rates",
       x = "Sales Channel",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per sales channel
preferred_seats_rate <- data %>%
  group_by(sales_channel) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between sales channel and preferred seats rates
ggplot(preferred_seats_rate, aes(x = sales_channel, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Sales Channel and Preferred Seats Rates",
       x = "Sales Channel",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### BOOKING COMPLETION

# Calculate proportion of 'Yes' for extra baggage per booking completion
extra_baggage_rate <- data %>%
  group_by(booking_complete) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between booking completion and extra baggage rates
ggplot(extra_baggage_rate, aes(x = booking_complete, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Booking Completion and Extra Baggage Rates",
       x = "Booking Completion",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per booking completion
in_flight_meals_rate <- data %>%
  group_by(booking_complete) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between booking completion and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = booking_complete, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Booking Completion and In-Flight Meals Rates",
       x = "Booking Completion",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per booking completion
preferred_seats_rate <- data %>%
  group_by(booking_complete) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between booking completion and preferred seats rates
ggplot(preferred_seats_rate, aes(x = booking_complete, y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Booking Completion and Preferred Seats Rates",
       x = "Booking Completion",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()


### RELATIONSHIP BETWEEN VARIABLES OF INTEREST ###

### EXTRA BAGGAGE

# Calculate proportion of 'Yes' for in-flight meals per extra baggage
in_flight_meals_rate <- data %>%
  group_by(wants_extra_baggage) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between extra baggage and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = as.factor(wants_extra_baggage), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Extra Baggage and In-Flight Meals Rates",
       x = "Extra Baggage",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per extra baggage
preferred_seats_rate <- data %>%
  group_by(wants_extra_baggage) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between extra baggage and preferred seats rates
ggplot(preferred_seats_rate, aes(x = as.factor(wants_extra_baggage), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Extra Baggage and Preferred Seats Rates",
       x = "Extra Baggage",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### IN-FLIGHT MEALS

# Calculate proportion of 'Yes' for extra baggage per in-flight meals
extra_baggage_rate <- data %>%
  group_by(wants_in_flight_meals) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between in-flight meals and extra baggage rates
ggplot(extra_baggage_rate, aes(x = as.factor(wants_in_flight_meals), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between In-Flight Meals and Extra Baggage Rates",
       x = "In-Flight Meals",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for preferred seats per in-flight meals
preferred_seats_rate <- data %>%
  group_by(wants_in_flight_meals) %>%
  summarise(yes_rate = mean(as.numeric(wants_preferred_seat) == 1))

# Plot the relationship between in-flight meals and preferred seats rates
ggplot(preferred_seats_rate, aes(x = as.factor(wants_in_flight_meals), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between In-Flight Meals and Preferred Seats Rates",
       x = "In-Flight Meals",
       y = "Proportion Wanting Preferred Seats") +
  theme_minimal()

### PREFERRED SEATS

# Calculate proportion of 'Yes' for extra baggage per preferred seats
extra_baggage_rate <- data %>%
  group_by(wants_preferred_seat) %>%
  summarise(yes_rate = mean(as.numeric(wants_extra_baggage) == 1))

# Plot the relationship between preferred seats and extra baggage rates
ggplot(extra_baggage_rate, aes(x = as.factor(wants_preferred_seat), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Preferred Seats and Extra Baggage Rates",
       x = "Preferred Seats",
       y = "Proportion Wanting Extra Baggage") +
  theme_minimal()

# Calculate proportion of 'Yes' for in-flight meals per preferred seats
in_flight_meals_rate <- data %>%
  group_by(wants_preferred_seat) %>%
  summarise(yes_rate = mean(as.numeric(wants_in_flight_meals) == 1))

# Plot the relationship between preferred seats and in-flight meals rates
ggplot(in_flight_meals_rate, aes(x = as.factor(wants_preferred_seat), y = yes_rate)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Relationship Between Preferred Seats and In-Flight Meals Rates",
       x = "Preferred Seats",
       y = "Proportion Wanting In-Flight Meals") +
  theme_minimal()


### MATRIX OF CORRELATION ###
#############################

# Load the corrplot package
library(corrplot)

# Selecting only the numerical features
numerical_data <- data[, sapply(data, is.numeric)]

# Calculating the correlation matrix
correlation_matrix <- cor(numerical_data)

# Plotting the correlation matrix
corrplot(correlation_matrix, method = "color", tl.col = "black")





