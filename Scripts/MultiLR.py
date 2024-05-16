import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import numpy as np
from pdpbox import pdp
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE


# Load the data
data = pd.read_csv('Data/cleaned_customer_booking.csv')

# Remove unneeded columns
data = data.drop(columns=['route', 'departure', 'arrival', 'booking_origin'])

# Combine the three binary target variables into a single multi-class label
data['combined_label'] = (data['wants_extra_baggage'].astype(str) +
                            data['wants_preferred_seat'].astype(str) +
                            data['wants_in_flight_meals'].astype(str))

# Prepare categorical variables with OneHotEncoder
categorical_vars = ['sales_channel', 'trip_type', 'flight_day', 'continent']
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_vars)], remainder='passthrough')

# Preprocess features
X = ct.fit_transform(data.drop(columns=['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'combined_label']))
y = data['combined_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the LogisticRegression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X_train, y_train)

### TEST SET ###

# Predict on the test data
y_test_pred = model.predict(X_test)
y_test_pred_prob = model.predict_proba(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_test_pred)
bal_acc = balanced_accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
cr = classification_report(y_test, y_test_pred)
print("Test Set Accuracy:", acc)
print("Test Set Balanced Accuracy:", bal_acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

# Calculate ROC AUC for each class
y_test_binarized = label_binarize(y_test, classes=model.classes_)
roc_auc = {}
for i in range(y_test_binarized.shape[1]):
  fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_pred_prob[:, i])
roc_auc[model.classes_[i]] = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc[model.classes_[i]]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic - Class {model.classes_[i]}')
plt.legend(loc="lower right")
plt.show()

### TRAINING SET ###

# Predict on the training data
y_train_pred = model.predict(X_train)
y_train_pred_prob = model.predict_proba(X_train)

# Evaluate the model
train_acc = accuracy_score(y_train, y_train_pred)
train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
train_cm = confusion_matrix(y_train, y_train_pred)
train_cr = classification_report(y_train, y_train_pred)
print("Training Set Accuracy:", train_acc)
print("Training Set Balanced Accuracy:", train_bal_acc)
print("Confusion Matrix on Training Data:\n", train_cm)
print("Classification Report on Training Data:\n", train_cr)


### VARIABLE IMPORTANCE ###
############################

# Calculate feature importances using permutation importance
importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=2022, n_jobs=-1)

# Create a DataFrame to display feature importances
feature_names = ct.get_feature_names_out()
importance_df = pd.DataFrame(data={
    'Feature': feature_names,
    'Importance': importance.importances_mean
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances for Logistic Regression')
plt.gca().invert_yaxis()
plt.show()


