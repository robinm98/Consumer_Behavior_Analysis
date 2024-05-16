import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

def create_model(input_dim, activation='relu', layers=2, dropout_rate=0.5, num_classes=8, nodes=128, l2_penalty=0.01):
    model = Sequential()
    model.add(Dense(nodes, activation=activation, input_dim=input_dim, kernel_regularizer=l2(l2_penalty)))
    model.add(Dropout(dropout_rate))
    for _ in range(1, layers):
        model.add(Dense(nodes, activation=activation, kernel_regularizer=l2(l2_penalty)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
data = pd.read_csv('Data/cleaned_customer_booking.csv')
data.drop(columns=['route', 'booking_origin', 'departure', 'arrival'], inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.difference(
    ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
)

# Combine labels into a single feature and convert to numeric
data['combined_label'] = pd.factorize(data['wants_extra_baggage'].astype(str) + 
                                      data['wants_in_flight_meals'].astype(str) + 
                                      data['wants_preferred_seat'].astype(str))[0]

# Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = preprocessor.fit_transform(data.drop(columns=['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'combined_label']))
y = to_categorical(data['combined_label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train the PRUNED best model found with GridSearchCV
best_model_params = {
    'activation': 'relu',
    'layers': 2,
    'dropout_rate': 0.5,
    'nodes': 128,
    'l2_penalty': 0.01,
}

model = create_model(input_dim=X_train.shape[1], num_classes=y.shape[1], **best_model_params)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=20, 
          verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

### TEST SET ###

# Predict on the test data
y_test_pred_prob = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Evaluate the model
test_acc = accuracy_score(y_test_classes, y_test_pred)
test_bal_acc = balanced_accuracy_score(y_test_classes, y_test_pred)
test_cm = confusion_matrix(y_test_classes, y_test_pred)
test_cr = classification_report(y_test_classes, y_test_pred)
print("Test Set Accuracy:", test_acc)
print("Test Set Balanced Accuracy:", test_bal_acc)
print("Confusion Matrix:\n", test_cm)
print("Classification Report:\n", test_cr)

### TRAINING SET ###

# Predict on the training data
y_train_pred_prob = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred_prob, axis=1)
y_train_classes = np.argmax(y_train, axis=1)

# Evaluate the model
train_acc = accuracy_score(y_train_classes, y_train_pred)
train_bal_acc = balanced_accuracy_score(y_train_classes, y_train_pred)
train_cm = confusion_matrix(y_train_classes, y_train_pred)
train_cr = classification_report(y_train_classes, y_train_pred)
print("Training Set Accuracy:", train_acc)
print("Training Set Balanced Accuracy:", train_bal_acc)
print("Confusion Matrix on Training Data:\n", train_cm)
print("Classification Report on Training Data:\n", train_cr)

### Permutation Importance ###
##############################

class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=1, verbose=0)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Wrap the trained model for permutation importance
wrapper = KerasModelWrapper(model)

# Calculate permutation importance
result = permutation_importance(wrapper, X, np.argmax(y, axis=1), n_repeats=10, random_state=42, n_jobs=-1)

# Create a DataFrame to hold the feature importances
importance_df = pd.DataFrame(data={
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': result.importances_mean
})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(14, 12))  # Increased figure height
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Neural Network')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
plt.subplots_adjust(left=0.4)  # Adjust left margin to make room for feature names
plt.show()
