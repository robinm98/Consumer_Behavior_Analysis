import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import numpy as np
from matplotlib import pyplot as plt

# Load the data
data = pd.read_csv('Data/cleaned_customer_booking.csv')

# Remove unneeded columns
data = data.drop(columns=['route', 'departure', 'arrival', 'booking_origin'])

# Combine labels into a single feature and convert to numeric
data['combined_label'] = pd.factorize(data['wants_extra_baggage'].astype(str) + 
                                      data['wants_in_flight_meals'].astype(str) + 
                                      data['wants_preferred_seat'].astype(str))[0]

# Define categorical and numerical columns
categorical_vars = ['sales_channel', 'trip_type', 'flight_day', 'continent']
numerical_vars = ['flight_duration', 'purchase_lead', 'num_passengers', 'length_of_stay', 'flight_hour']

# Preprocess features with ColumnTransformer
ct = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(), categorical_vars)
], remainder='passthrough')

# Define X and y
X = data.drop(columns=['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'combined_label', 'booking_complete'])
y = data['combined_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Ensure the same preprocessing is applied
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Reconstruct DataFrame for consistency
X_train = pd.DataFrame(X_train, columns=ct.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=ct.get_feature_names_out())

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the RandomForest model using the specified parameters
model = RandomForestClassifier(random_state=123, n_estimators=50, max_features=None, min_samples_split=10, min_samples_leaf=5)

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
acc = accuracy_score(y_test, predictions)
bal_acc = balanced_accuracy_score(y_test, predictions)
print("Test Set Accuracy:", acc)
print("Test Set Balanced Accuracy:", bal_acc)
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
model_pruned = RandomForestClassifier(random_state=123, n_estimators=50, max_features=None, min_samples_split=10, min_samples_leaf=5, max_depth=6)

# Fit the pruned model
model_pruned.fit(X_train, y_train)

### TEST SET FOR PRUNED MODEL ###

# Predict on the test data
predictions_pruned = model_pruned.predict(X_test)

# Predict probabilities for ROC and AUC calculations for pruned model
probs_pruned = model_pruned.predict_proba(X_test)

# Calculate ROC curve and AUC for each class for pruned model
fpr_pruned, tpr_pruned, roc_auc_pruned = {}, {}, {}
for i in range(y_test_binarized.shape[1]):
    fpr_pruned[i], tpr_pruned[i], _ = roc_curve(y_test_binarized[:, i], probs_pruned[:, i])
    roc_auc_pruned[i] = auc(fpr_pruned[i], tpr_pruned[i])

# Plot ROC curve for each class for pruned model
plt.figure()
for i in range(y_test_binarized.shape[1]):
    plt.plot(fpr_pruned[i], tpr_pruned[i], label=f'Class {i} (area = {roc_auc_pruned[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Pruned Model')
plt.legend(loc="lower right")
plt.show()

# Print the confusion matrix for the pruned model
print("Confusion Matrix for Pruned Model:")
print(confusion_matrix(y_test, predictions_pruned))

# Print a classification report for the pruned model
print("Classification Report for Pruned Model:")
print(classification_report(y_test, predictions_pruned))
print("Test Set Accuracy:", accuracy_score(y_test, predictions_pruned))
print("Test set Balanced Accuracy:", balanced_accuracy_score(y_test, predictions_pruned))

# AUC for each class for pruned model
for i in range(y_test_binarized.shape[1]):
    print(f"Class {i} AUC (Pruned): {roc_auc_pruned[i]:.2f}")

class_report_pruned = classification_report(y_test, predictions_pruned)

# save the accuracy, balanced accuracy, precision, and recall to a file
results = pd.DataFrame({
    'Accuracy': [accuracy_score(y_test, predictions_pruned)],
    'Balanced Accuracy': [balanced_accuracy_score(y_test, predictions_pruned)],
    'Precision': [class_report_pruned.split()[5]],
    'Recall': [class_report_pruned.split()[6]]
    })
results.to_csv('Data/RF_results.csv', index=False)

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


### VARIABLE IMPORTANCE ###
###########################

# Compute feature importance using permutation importance
result = permutation_importance(model_pruned, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Create a DataFrame to hold the feature importances
importance_df = pd.DataFrame(data={
    'Feature': feature_names,
    'Importance': result.importances_mean
})
print(feature_names)

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importance_df)

# Plot feature importances
plt.figure(figsize=(12, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Pruned Random Forest')
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.4) # Adjust left margin to make room for feature names
plt.show()

### Partial Dependence Plot ###
###############################

### Numerical Features ###

def compute_pdp(model_pruned, X, feature_index, values):
    pdp = []
    X_temp = X.copy()
    for value in values:
        X_temp.iloc[:, feature_index] = value
        preds = model_pruned.predict_proba(X_temp)
        pdp.append(np.mean(preds, axis=0))
    return np.array(pdp)

# Get feature names and indices
feature_names = ct.get_feature_names_out()
feature_indices = {name: idx for idx, name in enumerate(feature_names)}

# Generate PDP for each numerical feature
for feature in numerical_vars:
    feature_name_transformed = f'remainder__{feature}'  # Adjust based on your pipeline's naming
    if feature_name_transformed in feature_indices:
        feature_index = feature_indices[feature_name_transformed]
        values = np.linspace(X_test.iloc[:, feature_index].min(), X_test.iloc[:, feature_index].max(), num=100)
        pdp = compute_pdp(model_pruned, X_test, feature_index, values)

        # Plot PDP for each class
        for i in range(pdp.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.plot(values, pdp[:, i], label=f'Class {i}')
            plt.xlabel(feature)
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence of {feature} for Class {i}')
            plt.legend()
            plt.show()
print("PDP generation complete.")

### Continent Feature ###

# Generate PDP for each one-hot encoded continent feature
for feature in one_hot_encoded_continents:
    if feature in feature_indices:
        feature_index = feature_indices[feature]
        # Values for one-hot encoded feature: 0 (not this category) and 1 (this category)
        values = [0, 1]
        pdp = compute_pdp_categorical(model_pruned, X_test, feature_index, values)

        # Ensure PDP has data and plot it
        if pdp.size > 0:
            for i in range(pdp.shape[1]):
                plt.figure(figsize=(8, 6))
                plt.plot(values, pdp[:, i], marker='o', linestyle='-', label=f'Class {i}')
                plt.xlabel(feature)
                plt.ylabel('Partial Dependence')
                plt.title(f'Partial Dependence of {feature} for Class {i}')
                plt.legend()
                plt.show()

print("PDP generation complete.")
