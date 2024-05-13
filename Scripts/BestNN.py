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

# Perform downsampling for 'wants_preferred_seat'
positive = data[data['wants_preferred_seat'] == 1]
negative = data[data['wants_preferred_seat'] == 0]
n_samples = min(len(positive), len(negative))
balanced_data = pd.concat([positive.sample(n_samples, random_state=42), negative.sample(n_samples, random_state=42)])

# Perform downsampling for 'wants_extra_baggage'
positive = balanced_data[balanced_data['wants_extra_baggage'] == 1]
negative = balanced_data[balanced_data['wants_extra_baggage'] == 0]
n_samples = min(len(positive), len(negative))
balanced_data = pd.concat([positive.sample(n_samples, random_state=42), negative.sample(n_samples, random_state=42)])

# Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = preprocessor.fit_transform(balanced_data.drop(['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals'], axis=1))
y = balanced_data[['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model with the best parameters
model = create_model(input_dim=X_train.shape[1])
model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics for each label
labels = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
metrics = {}

for i, label in enumerate(labels):
    acc = accuracy_score(y_test[:, i], y_pred[:, i])
    bal_acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
    auc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
    metrics[label] = {'Accuracy': acc, 'Balanced Accuracy': bal_acc, 'ROC AUC': auc}
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    cr = classification_report(y_test[:, i], y_pred[:, i])
    print(f"Metrics for {label}:")
    print("Accuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("ROC AUC:", auc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)
