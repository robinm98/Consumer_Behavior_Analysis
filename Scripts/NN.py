import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import BinaryAccuracy

def create_model(input_dim, layers=1, activation='relu', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, activation=activation, input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    for _ in range(1, layers):
        model.add(Dense(64, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

X = preprocessor.fit_transform(data.drop(['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals'], axis=1))
y = data[['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid
param_grid = {
    'epochs': [20, 50],
    'batch_size': [16, 32],
    'layers': [1, 2],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.5, 0.6]
}

best_score = 0
best_params = {}

for params in ParameterGrid(param_grid):
    # Separate model parameters and training parameters
    model_params = {key: params[key] for key in params if key in ['layers', 'activation', 'dropout_rate']}
    train_params = {key: params[key] for key in params if key in ['epochs', 'batch_size']}
    
    model = create_model(input_dim=X_train.shape[1], **model_params)
    model.fit(X_train, y_train, **train_params, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)[1]  # Get accuracy
    if score > best_score:
        best_score = score
        best_params = params

print("Best score: {:.2f}".format(best_score))
print("Best parameters:", best_params)