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
data_num_scaled <- as.data.frame(scale(data_num))

# Sample 10000 rows to find optimal number of clusters
data_sample <- data_num_scaled[sample(nrow(data_num_scaled), 10000), ] 

### Find the optimal number of clusters

# WSS
fviz_nbclust(data_sample, kmeans, method = "wss", verbose = FALSE) # 3 clusters

# Silhouette
fviz_nbclust(data_sample, kmeans, method = "silhouette", verbose = FALSE) # 2 clusters

# Gap statistic
fviz_nbclust(data_sample, kmeans, method = "gap_stat", verbose = FALSE) # 1 clusters

### Perform K-means clustering with 3 clusters to fall between the optimal number of clusters

# K-means clustering
km_model <- kmeans(data_num_scaled, centers = 3, nstart = 25)
km_model

### K-means Cluster Analysis ###
###############################

### Cluster profile ###

# Add cluster assignments to data
data_num$cluster <- km_model$cluster

# Calculate mean for each cluster and each variable
cluster_profiles <- data_num |>
  group_by(cluster) |>
  summarise_all(mean, na.rm = TRUE)   # Change mean to median or another function if more appropriate

print(cluster_profiles)

### Number of observations in each cluster
table(data_num$cluster)

### PCA Analysis ###

# Add cluster assignments to scaled data
data_num_scaled$cluster <- km_model$cluster

# Perform PCA on the scaled data
pca_res <- prcomp(data_num_scaled[, -ncol(data_num_scaled)], scale. = TRUE)
summary(pca_res)

# Extract PCA loadings
loadings <- data.frame(Variable = rownames(pca_res$rotation), pca_res$rotation)

# Scale loadings for better visualization
loadings$PC1 <- loadings$PC1 * max(pca_res$x[, 1])
loadings$PC2 <- loadings$PC2 * max(pca_res$x[, 2])

# Create a data frame for plotting
pca_data <- data.frame(pca_res$x, cluster = km_model$cluster)

# Plotting the first two principal components with loadings
ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(cluster))) +
  geom_point(alpha = 0.5) +
  geom_segment(data = loadings, aes(x = 0, y = 0, xend = PC1, yend = PC2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "black") +
  geom_text(data = loadings, aes(x = PC1, y = PC2, label = Variable), 
            vjust = 1, hjust = 1, color = "black") +
  labs(title = "PCA Plot of Clusters with Loadings",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster") +
  coord_cartesian(xlim = c(-17, 12), ylim = c(-3, 10)) 

### Boxplot of all numerical variables by cluster ###

# Melting data for ggplot2 usage
data_melted <- reshape2::melt(data_num, id.vars = "cluster")

# Filter the data for purchase_lead and length_of_stay conditions
data_filtered <- data_melted %>%
  filter((variable != "purchase_lead" | value < 300) &
           (variable != "length_of_stay" | value < 50))

# Boxplot of filtered variables by cluster
ggplot(data_filtered, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
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
baggage_cluster_table <- table(data$wants_extra_baggage, data_num$cluster)

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
  labs(x = "Cluster", y = "Percentage", fill = "Wants Extra Bagage") +
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

# Create dummy variables for categorical columns
data_dummy <- dummy_cols(data, select_columns = c("sales_channel", "trip_type", "booking_complete"))

# Remove categorical variables
data_dummy <- data_dummy[, -c(2:3, 7:9, 15:17)]

# Sample 10000 rows to find optimal number of clusters
data_dummy_sample <- data_dummy[sample(nrow(data_dummy), 10000), ]

# Specify numerical variables
num_vars <- c(1:4, 8)

# Scale numerical variables
scaled_num_vars <- scale(data_dummy_sample[, num_vars])

# Combine scaled numerical variables with dummy variables
data_dummy_scaled <- cbind(data_dummy_sample[, -num_vars], scaled_num_vars)


### Find the optimal number of clusters

# WSS
fviz_nbclust(data_dummy_scaled, pam, method = "wss", verbose = FALSE) # 4 clusters

# Silhouette
fviz_nbclust(data_dummy_scaled, pam, method = "silhouette", verbose = FALSE) # 3 clusters

# Gap statistic
fviz_nbclust(data_dummy_scaled, pam, method = "gap_stat", verbose = FALSE) # 3 clusters

### Perform PAM clustering with 4 clusters to fall between the optimal number of clusters

# PAM clustering
pam_model <- pam(data_dummy_sample, k = 4)
pam_model

### PAM Cluster Analysis ###
############################

### Cluster profile ###

# Add cluster assignments to non scaled data
data_dummy_sample$cluster <- pam_model$clustering

# Calculate mean for each cluster and each variable
cluster_profiles <- data_dummy_sample |>
  group_by(cluster) |>
  summarise_all(mean, na.rm = TRUE)  # Change mean to median or another function if more appropriate

print(cluster_profiles)

### PCA Analysis ###

# Add cluster assignments to scaled data
data_dummy_scaled$cluster <- pam_model$clustering

# Perform PCA on the scaled data
pca_res <- prcomp(data_dummy_scaled[, -ncol(data_dummy_scaled)], scale. = TRUE)
summary(pca_res)

# Plot PCA
biplot(pca_res, scale = 0, cex = 0.6)

# Create a data frame for plotting
pca_data <- data.frame(pca_res$x, cluster = pam_model$clustering)

# Plotting the first two principal components
ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "PCA Plot of Clusters",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster")

### Boxplot of all numerical variables by cluster ###

# Melting data for ggplot2 usage
data_melted <- reshape2::melt(data_dummy_sample, id.vars = "cluster")

# Filter melted data to include only numerical variables
data_melted_numeric <- data_melted %>%
  filter(variable %in% names(data_dummy_sample)[c(1:4, 8)])

# Boxplot of numerical variables by cluster
ggplot(data_melted_numeric, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free_y") +  
  theme_minimal() +
  labs(title = "Distribution of Numerical Features Across Clusters",
       x = "Cluster",
       y = "Value",
       fill = "Cluster")

### Categorical variables by cluster ###

data <- data[sample(nrow(data), 10000), ]

# sales_channel
table(data$sales_channel, data_dummy_sample$cluster)

# trip_type
table(data$trip_type, data_dummy_sample$cluster)

# flight_day
table(data$flight_day, data_dummy_sample$cluster)

# continent
table(data$continent, data_dummy_sample$cluster)

# booking_complete
table(data$booking_complete, data_dummy_sample$cluster)

### Proportion of interest variables by cluster ###

# wants_extra_baggage

# Create a table of counts
baggage_cluster_table <- table(data_dummy_sample$wants_extra_baggage, data_dummy_sample$cluster)

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
  labs(x = "Cluster", y = "Percentage", fill = "Wants Extra Baggage") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +  # Customize fill colors if needed
  theme_minimal()

# wants_in_flight_meal

# Table of counts
meal_cluster_table <- table(data_dummy_sample$wants_in_flight_meal, data_dummy_sample$cluster)

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
seat_cluster_table <- table(data_dummy_sample$wants_preferred_seat, data_dummy_sample$cluster)

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

