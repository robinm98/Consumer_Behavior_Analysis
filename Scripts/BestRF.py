import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
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

# Ensure labels are combined into a single feature and converted to numeric
data['combined_label'] = pd.factorize(data['wants_extra_baggage'].astype(str) + 
                                      data['wants_in_flight_meals'].astype(str) + 
                                      data['wants_preferred_seat'].astype(str))[0]

# Append combined_label to processed data
data_processed['combined_label'] = data['combined_label']

# Prepare training and testing data
X = data_processed.drop(columns=['remainder__wants_extra_baggage',
       'remainder__wants_preferred_seat', 'remainder__wants_in_flight_meals','combined_label'], errors='ignore')
y = data_processed['combined_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the RandomForest model using the specified parameters
model = RandomForestClassifier(random_state=123, n_estimators=250, max_features=None, min_samples_split=10, min_samples_leaf=5)

# Fit the model
model.fit(X_train, y_train)

### TEST SET ###

# Predict on the test data
predictions = model.predict(X_test)

# Predict probabilities for ROC and AUC calculations
probs = model.predict_proba(X_test)

# Binarize the output for the multi-class case
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(y_test_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(y_test_binarized.shape[1]):
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
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

### TRAINING SET ###

# Predict on the training data using the best model found by grid search
train_predictions = model.predict(X_train)

# Compute confusion matrix for the training data
train_conf_matrix = confusion_matrix(y_train, train_predictions)

# Print the confusion matrix for the training data
print("Confusion Matrix on Training Data:")
print(train_conf_matrix)

# Print a classification report for the training data
train_class_report = classification_report(y_train, train_predictions)
print("Classification Report on Training Data:")
print(train_class_report)
# --> CLEAR SIGN OF OVERFITTING


### RF PRUNING ###
##################

# Prune the model (no package to prune RF like CART)
model_pruned = RandomForestClassifier(random_state=123, n_estimators=250, max_features='sqrt', min_samples_split=50, min_samples_leaf=40, max_depth=5)

# Fit the pruned model
model_pruned.fit(X_train, y_train)

### TEST SET ###

# Predict on the test data
predictions_pruned = model_pruned.predict(X_test)

# Compute confusion matrix for the pruned model
conf_matrix_pruned = confusion_matrix(y_test, predictions_pruned)

# Print the confusion matrix for the pruned model
print("Confusion Matrix for Pruned Model:")
print(conf_matrix_pruned)

# Print a classification report for the pruned model
class_report_pruned = classification_report(y_test, predictions_pruned)
print("Classification Report for Pruned Model:")
print(class_report_pruned)
print("Test Set Accuracy:", accuracy_score(y_test, predictions_pruned))
print("Test set Balanced Accuracy:", balanced_accuracy_score(y_test, predictions_pruned))

### TRAINING SET ###

# Predict on the training data using the pruned model
train_predictions_pruned = model_pruned.predict(X_train)

# Compute confusion matrix for the training data using the pruned model
train_conf_matrix_pruned = confusion_matrix(y_train, train_predictions_pruned)

# Print the confusion matrix for the training data using the pruned model
print("Confusion Matrix on Training Data for Pruned Model:")
print(train_conf_matrix_pruned)

# Print a classification report for the training data using the pruned model
train_class_report_pruned = classification_report(y_train, train_predictions_pruned)
print("Classification Report on Training Data for Pruned Model:")
print(train_class_report_pruned)
print("Training Set Accuracy:", accuracy_score(y_train,train_predictions_pruned))
print("Training Set Balanced Accuracy:", balanced_accuracy_score(y_train,train_predictions_pruned))
