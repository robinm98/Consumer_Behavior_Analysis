import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import BinaryAccuracy

def create_model(input_dim, layers=1, activation='relu', dropout_rate=0.5):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Use Input layer to specify the shape
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(1, layers):
        model.add(Dense(64, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Single output for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    return model

# Load and prepare data
data = pd.read_csv('Data/cleaned_customer_booking.csv')
data.drop(columns=['route', 'booking_origin', 'departure', 'arrival'], inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.difference(
    ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Define target variables
target_vars = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']

# Parameter grid
param_grid = {
    'epochs': [20, 50],
    'batch_size': [16, 32],
    'layers': [1, 2],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.5, 0.6]
}

# Function to perform grid search for each target variable
def perform_grid_search(target_var):
    print(f"Performing grid search for: {target_var}")
    
    # Prepare data for the target variable
    X = preprocessor.fit_transform(data.drop(target_vars, axis=1))
    y = data[target_var].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_score = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        # Separate model parameters and training parameters
        model_params = {key: params[key] for key in params if key in ['layers', 'activation', 'dropout_rate']}
        train_params = {key: params[key] for key in params if key in ['epochs', 'batch_size']}
        
        model = create_model(input_dim=X_train.shape[1], **model_params)
        model.fit(X_train, y_train, **train_params, verbose=1)
        score = model.evaluate(X_test, y_test, verbose=1)[1]  # Get accuracy
        if score > best_score:
            best_score = score
            best_params = params

    print(f"Best score for {target_var}: {best_score:.2f}")
    print(f"Best parameters for {target_var}: {best_params}")
    return best_score, best_params

# Perform grid search for each target variable
best_scores = {}
best_parameters = {}

for target_var in target_vars:
    best_score, best_params = perform_grid_search(target_var)
    best_scores[target_var] = best_score
    best_parameters[target_var] = best_params

print("Best scores:", best_scores)
print("Best parameters:", best_parameters)

