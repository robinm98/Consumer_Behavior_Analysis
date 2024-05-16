import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
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

# Prepare cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
num_classes = y.shape[1]

# Store results
val_accuracies = []
val_bal_accuracies = []
train_accuracies = []
train_bal_accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=123)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model = create_model(input_dim=X_train.shape[1], num_classes=num_classes, activation='relu', layers=2, dropout_rate=0.5, nodes=128, l2_penalty=0.01)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Predict on the validation data
    y_val_pred_prob = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
    val_acc = accuracy_score(y_val_classes, y_val_pred)
    val_bal_acc = balanced_accuracy_score(y_val_classes, y_val_pred)
    
    val_accuracies.append(val_acc)
    val_bal_accuracies.append(val_bal_acc)

    # Predict on the training data
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)
    
    train_acc = accuracy_score(y_train_classes, y_train_pred)
    train_bal_acc = balanced_accuracy_score(y_train_classes, y_train_pred)
    
    train_accuracies.append(train_acc)
    train_bal_accuracies.append(train_bal_acc)

print("Validation Accuracies:", val_accuracies)
print("Mean Validation Accuracy:", np.mean(val_accuracies))
print("Mean Validation Balanced Accuracy:", np.mean(val_bal_accuracies))
print("\nTraining Accuracies:", train_accuracies)
print("Mean Training Accuracy:", np.mean(train_accuracies))
print("Mean Training Balanced Accuracy:", np.mean(train_bal_accuracies))

