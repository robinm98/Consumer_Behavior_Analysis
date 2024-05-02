#### MLBA Project - Modeling ####
#################################

# Load the required libraries

library(rpart)
library(caret)
library(randomForest)

# Load the data
data <- read.csv(here("Data", "customer_booking.csv"))

### 3 models : Logistic Regression, Decision Tree, Random Forest
### 3 variable of interest : wants_extra_luggage, wants_in_flight_meal, wants_prefered_seat

