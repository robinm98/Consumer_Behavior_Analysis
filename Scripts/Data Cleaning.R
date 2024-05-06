#### MLBA Project - Data Cleaning ####
######################################

# Load the required libraries
library(dplyr)
library(tidyr)
library(tidyverse)
library(readr)
library(here)
library(countrycode)

# Load the data
data <- read.csv(here("Data", "raw_customer_booking.csv"))
                     
# Display the first few rows of the data
head(data)

# Provide a summary of the dataset
summary(data)

### Missing Values ###
######################

# Drop rows with missing values
data <- data %>% drop_na()

# Check for missing values
missing_values <- data %>% summarise_all(funs(sum(is.na(.))))
missing_values # <- no missing values

### Data Cleaning ###
#####################

# Create a column 'continent' based on 'booking_origin'
data$continent <- countrycode(data$booking_origin, "country.name", "continent")

# Ensure consistent encoding because of special characters in 'Réunion'
data$booking_origin <- iconv(data$booking_origin, from = "latin1", to = "UTF-8")

# Replace "R�union" with "Reunion" in the booking_origin column
data <- data %>%
  mutate(booking_origin = gsub("R.union", "Reunion", booking_origin))

# Now apply the continent mapping for "Reunion"
data <- data %>%
  mutate(continent = ifelse(booking_origin == "Reunion", "Africa", continent))

# Set continent to 'Unknown' for missing values in 'booking_origin'
data$continent[is.na(data$continent)] <- "Unknown"

# Separate column 'route' into 'departure' and 'arrival'
data$departure <- substr(data$route, 1, 3)  # Extract the first 3 characters for departure
data$arrival <- substr(data$route, 4, 6)    # Extract the last 3 characters for arrival

# Convert categorical variables to factors
data$sales_channel <- as.factor(data$sales_channel)
data$trip_type <- as.factor(data$trip_type)
data$route <- as.factor(data$route)
data$flight_day <- factor(data$flight_day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
data$booking_origin <- as.factor(data$booking_origin)
data$departure <- as.factor(data$departure)
data$arrival <- as.factor(data$arrival)
data$continent <- as.factor(data$continent)

# Convert wants_extra_baggage to numeric : other 'wants' already numeric
data$wants_extra_baggage <- as.numeric(as.character(data$wants_extra_baggage))


# Save the cleaned data
write.csv(data, here("Data", "cleaned_customer_booking.csv"), row.names = FALSE)