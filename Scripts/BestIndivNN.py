import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score

def create_model(input_dim, activation='relu', layers=2, dropout_rate=0.6):
    model = Sequential()
    model.add(Dense(64, activation=activation, input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    for _ in range(1, layers):
        model.add(Dense(64, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Single output for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
data = pd.read_csv('Data/cleaned_customer_booking.csv')
data.drop(columns=['route', 'booking_origin', 'departure', 'arrival'], inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.difference(
    ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
)

# Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Function to train and evaluate a model for a specific target variable
def train_and_evaluate_nn(target_var, other_vars):
    print(f"Training and evaluating model for: {target_var}")

    # Perform downsampling for the target variable
    positive = data[data[target_var] == 1]
    negative = data[data[target_var] == 0]
    n_samples = min(len(positive), len(negative))
    balanced_data = pd.concat([positive.sample(n_samples, random_state=42), negative.sample(n_samples, random_state=42)])

    # Preprocess features
    X = preprocessor.fit_transform(balanced_data.drop(columns=[target_var] + other_vars))
    y = balanced_data[target_var].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)

    ### TEST SET ###

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Metrics for {target_var}:")
    print("Accuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("ROC AUC:", auc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    ### TRAINING SET ###

    # Predictions
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)

    # Calculate metrics for each label
    acc_train = accuracy_score(y_train, y_train_pred)
    bal_acc_train = balanced_accuracy_score(y_train, y_train_pred)
    auc_train = roc_auc_score(y_train, y_train_pred_prob)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cr_train = classification_report(y_train, y_train_pred)
    print(f"Training Metrics for {target_var}:")
    print("Accuracy:", acc_train)
    print("Balanced Accuracy:", bal_acc_train)
    print("ROC AUC:", auc_train)
    print("Confusion Matrix:\n", cm_train)
    print("Classification Report:\n", cr_train)

# List of target variables to model independently
target_vars = {
    'wants_extra_baggage': ['wants_preferred_seat', 'wants_in_flight_meals'],
    'wants_preferred_seat': ['wants_extra_baggage', 'wants_in_flight_meals'],
    'wants_in_flight_meals': ['wants_extra_baggage', 'wants_preferred_seat']
}

# Train and evaluate the model for each target variable
for target_var, other_vars in target_vars.items():
    train_and_evaluate_nn(target_var, other_vars)
