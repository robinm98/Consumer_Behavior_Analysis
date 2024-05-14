import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

# Load the data
data = pd.read_csv('Data/cleaned_customer_booking.csv')

# Remove unneeded columns
data = data.drop(columns=['route', 'departure', 'arrival', 'booking_origin'])

# Prepare categorical variables with OneHotEncoder
categorical_vars = ['sales_channel', 'trip_type', 'flight_day', 'continent']
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_vars)], remainder='passthrough')
data_processed = ct.fit_transform(data)

# Reconstruct dataframe after encoding
data_processed = pd.DataFrame(data_processed, columns=ct.get_feature_names_out())

# Make sure you are addressing the correct column names based on output
data_processed['combined_label'] = pd.factorize(data['wants_extra_baggage'].astype(str) + 
                                                data['wants_in_flight_meals'].astype(str) + 
                                                data['wants_preferred_seat'].astype(str))[0]

# Prepare training and testing data

X = data_processed.drop(columns=['remainder__wants_extra_baggage',
       'remainder__wants_preferred_seat', 'remainder__wants_in_flight_meals', 'combined_label'], errors='ignore')
y = data_processed['combined_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the RandomForest model with fewer trees
model = RandomForestClassifier(random_state=123, n_estimators=50)  # Reduced number of trees

# Define parameter grid focusing on fewer trees and tree complexity
param_grid = {
    'max_features': ['sqrt', 'log2', None],  # Features considered for splitting at each leaf
    'min_samples_split': [10, 20],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [5, 10]  # Minimum number of samples required to be at a leaf node
}

# GridSearchCV for parameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Predict probabilities for the test set
probs = grid_search.predict_proba(X_test)

# Compute ROC curve and AUC for each class (assuming a multiclass classification problem)
# Binarize the output
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate ROC curve and AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific class or aggregate
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print confusion matrix and classification report
predictions = grid_search.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Best Parameters:", grid_search.best_params_)