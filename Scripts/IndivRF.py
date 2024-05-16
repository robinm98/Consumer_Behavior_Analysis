import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, roc_curve, auc
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

# Function to train, perform GridSearchCV, and evaluate a Random Forest model for a specific target variable
def train_and_evaluate_rf(target_var, other_vars):
    print(f"Training and evaluating model for: {target_var}")
    
    # Define features and target
    X = data.drop(columns=other_vars + [target_var])
    X_processed = ct.fit_transform(X)
    y = data[target_var]
    
    # Prepare training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=123)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=123)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Define the RandomForest model with fewer trees
    model = RandomForestClassifier(random_state=123, n_estimators=50)
    
    # Define parameter grid focusing on fewer trees and tree complexity
    param_grid = {
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }
    
    # GridSearchCV for parameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Model:", best_model)
    
    # Train the best model
    best_model.fit(X_train, y_train)
    
    # Predict on the training set
    y_train_pred = best_model.predict(X_train)
    
    # Predict on the test set
    y_test_pred = best_model.predict(X_test)
    
    # Evaluate the model on the training set
    print("\nTraining Metrics:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred))
    print("\nAccuracy Score:", accuracy_score(y_train, y_train_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_train, y_train_pred))
    
    # Evaluate the model on the test set
    print("\nTest Metrics:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_test_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_test_pred))
    
    # Plot ROC Curve for the test set
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {target_var}')
    plt.legend(loc="lower right")
    plt.show()

# List of target variables to model independently
target_vars = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']

# Train and evaluate the model for each target variable
for target_var in target_vars:
    other_vars = [var for var in target_vars if var != target_var]
    train_and_evaluate_rf(target_var, other_vars)

