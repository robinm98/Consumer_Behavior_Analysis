#### MLBA Project - Unsupervised Learning ####
##############################################

# Load libraries
library(here)
library(ggplot2)
library(factoextra)
library(cluster)
library(dplyr)
library(GGally)
library(reshape2)
library(fastDummies)

# Load data
data <- read.csv(here("data", "cleaned_customer_booking.csv"))


### Clustering ###
##################

set.seed(123)

### K-means ###
###############

### Data Preprocessing ###
##########################

# Scale the numerical features
data_num <- data[, c(1, 4:6, 13)]
data_num_scaled <- scale(data_num) # Scale all the numerical features

# Sample 10000 rows to find optimal number of clusters
data_sample <- data_num_scaled[sample(nrow(data_num_scaled), 10000), ] 

### Find the optimal number of clusters

# WSS
fviz_nbclust(data_sample, kmeans, method = "wss", verbose = FALSE) # 6 clusters

# Silhouette
fviz_nbclust(data_sample, kmeans, method = "silhouette", verbose = FALSE) # 2 clusters

# Gap statistic
fviz_nbclust(data_sample, kmeans, method = "gap_stat", verbose = FALSE) # 1 clusters

### Perform K-means clustering with 3 clusters to fall between the optimal number of clusters

# K-means clustering
km_model <- kmeans(data_num_scaled, centers = 3, nstart = 25)
km_model

### K-means Cluster Analysis ###

### Cluster profile ###

# Add cluster assignments to data
data_num$cluster <- km_model$cluster

# Calculate mean for each cluster and each variable
cluster_profiles <- data_num |>
  group_by(cluster) |>
  summarise_all(mean, na.rm = TRUE)  # Change mean to median or another function if more appropriate

print(cluster_profiles)

### PCA Analysis ###

# Perform PCA on the scaled data
pca_res <- prcomp(data_num[, -ncol(data_num)], scale. = TRUE)
summary(pca_res)

# Plot PCA
biplot(pca_res, scale = 0, cex = 0.6)

# Create a data frame for plotting
pca_data <- data.frame(pca_res$x, cluster = km_model$cluster)

# Plotting the first two principal components
ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "PCA Plot of Clusters",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster")


### Boxplot of all numerical variables by cluster ###

# Melting data for ggplot2 usage
data_melted <- reshape2::melt(data_num, id.vars = "cluster")

# Boxplot of all variables by cluster using the correctly melted data
ggplot(data_melted, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free_y") +  # Ensure variable names are correct for faceting
  theme_minimal() +
  labs(title = "Distribution of Features Across Clusters",
       x = "Cluster",
       y = "Value",
       fill = "Cluster")

### Categorical variables by cluster ###

# sales_channel
table(data$sales_channel, data_num$cluster)

# trip_type
table(data$trip_type, data_num$cluster)

# flight_day
table(data$flight_day, data_num$cluster)

# continent
table(data$continent, data_num$cluster)

# booking_complete
table(data$booking_complete, data_num$cluster)

### Proportion of interest variables by cluster ###

# wants_extra_baggage

# Create a table of counts
baggage_cluster_table <- table(data$wants_preferred_seat, data_num$cluster)

# Convert counts to proportions
prop_cluster_table <- prop.table(baggage_cluster_table, margin = 2)

# Convert the table to a data frame for plotting
prop_cluster_df <- as.data.frame(prop_cluster_table)

# Calculate percentages within each cluster
prop_cluster_df <- prop_cluster_df %>%
  group_by(Var2) %>%
  mutate(percentage = Freq / sum(Freq) * 100)

# Convert factor column to character
prop_cluster_df$Var1 <- as.character(prop_cluster_df$Var1)

# Plot the proportions
ggplot(prop_cluster_df, aes(x = Var2, y = percentage, fill = Var1, group = Var1)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(x = "Cluster", y = "Percentage", fill = "Wants Preferred Seat") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +  # Customize fill colors if needed
  theme_minimal()

# wants_in_flight_meal

# Table of counts
meal_cluster_table <- table(data$wants_in_flight_meal, data_num$cluster)

# Convert counts to proportions
prop_cluster_table <- prop.table(meal_cluster_table, margin = 2)

# Convert the table to a data frame for plotting
prop_cluster_df <- as.data.frame(prop_cluster_table)

# Calculate percentages within each cluster
prop_cluster_df <- prop_cluster_df %>%
  group_by(Var2) %>%
  mutate(percentage = Freq / sum(Freq) * 100)

# Convert factor column to character
prop_cluster_df$Var1 <- as.character(prop_cluster_df$Var1)

# Plot the proportions
ggplot(prop_cluster_df, aes(x = Var2, y = percentage, fill = Var1, group = Var1)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(x = "Cluster", y = "Percentage", fill = "Wants In-Flight Meal") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +  # Customize fill colors if needed
  theme_minimal()

# wants_preferred_seat

# Table of counts
seat_cluster_table <- table(data$wants_preferred_seat, data_num$cluster)

# Convert counts to proportions
prop_cluster_table <- prop.table(seat_cluster_table, margin = 2)

# Convert the table to a data frame for plotting
prop_cluster_df <- as.data.frame(prop_cluster_table)

# Calculate percentages within each cluster
prop_cluster_df <- prop_cluster_df %>%
  group_by(Var2) %>%
  mutate(percentage = Freq / sum(Freq) * 100)

# Convert factor column to character
prop_cluster_df$Var1 <- as.character(prop_cluster_df$Var1)

# Plot the proportions
ggplot(prop_cluster_df, aes(x = Var2, y = percentage, fill = Var1, group = Var1)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(x = "Cluster", y = "Percentage", fill = "Wants Preferred Seat") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +  # Customize fill colors if needed
  theme_minimal()

### PAM ###
###########


### Data Preprocessing ###
##########################

# Transform categorical variable in dummy variables
data_dummy <- dummy_cols(data, select_columns = c("sales_channel", "trip_type", "continent", "booking_complete"))

# Remove categorical variables
data_dummy <- data_dummy[, -c(2:3, 7:9, 15:17)]

# Specify numerical variables
num_vars <- c(1, 4:6, 13)

# Scale numerical variables
scaled_num_vars <- scale(data_dummy[, num_vars])

# Combine scaled numerical variables with dummy variables
data_dummy_scaled <- cbind(data_dummy[, -num_vars], scaled_num_vars)

# Sample 10000 rows to find optimal number of clusters
data_dummy_sample <- data_dummy_scaled[sample(nrow(data_dummy_scaled), 10000), ]

### Find the optimal number of clusters

# WSS
fviz_nbclust(data_dummy_sample, pam, method = "wss", verbose = FALSE) # 4 clusters

# Silhouette
fviz_nbclust(data_dummy_sample, pam, method = "silhouette", verbose = FALSE) # 2 clusters

# Gap statistic
fviz_nbclust(data_dummy_sample, pam, method = "gap_stat", verbose = FALSE) # 3 clusters

### Perform PAM clustering with X clusters to fall between the optimal number of clusters

# PAM clustering
pam_model <- pam(data_dummy_scaled, k = 3)
pam_model

### PAM Cluster Analysis ###

### Cluster profile ###

# Add cluster assignments to data
data_dummy$cluster <- pam_model$clustering

# Calculate mean for each cluster and each variable
cluster_profiles <- data_dummy |>
  group_by(cluster) |>
  summarise_all(mean, na.rm = TRUE)  # Change mean to median or another function if more appropriate

print(cluster_profiles)

