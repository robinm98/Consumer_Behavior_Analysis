
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

def create_model(input_dim, activation='relu', layers=1, nodes=64, dropout_rate=0.5, num_classes=8):
    model = Sequential()
    model.add(Dense(nodes, activation=activation, input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    for _ in range(1, layers):
        model.add(Dense(nodes, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
data = pd.read_csv('../Data/cleaned_customer_booking.csv')
data.drop(columns=['route', 'booking_origin', 'departure', 'arrival'], inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.difference(
    ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
)

# Combine labels into a single feature and convert to numeric
balanced_data['combined_label'] = pd.factorize(balanced_data['wants_extra_baggage'].astype(str) + 
                                               balanced_data['wants_in_flight_meals'].astype(str) + 
                                               balanced_data['wants_preferred_seat'].astype(str))[0]

# Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = preprocessor.fit_transform(balanced_data.drop(columns=['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'combined_label']))
y = to_categorical(balanced_data['combined_label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Wrap the model using KerasClassifier
num_classes = y.shape[1]
model = KerasClassifier(build_fn=create_model, input_dim=X_train.shape[1], verbose=1, activation='relu', layers=1, nodes=64, dropout_rate=0.5, num_classes=8)

# Define parameter grid
param_grid = {
    'layers': [1, 2, 3],
    'nodes': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'epochs': [20],
    'batch_size': [32],
    'dropout_rate': [0.5]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Model:", best_model)

### TEST SET ###

# Predict on the test set
y_pred_prob = best_model.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate metrics
acc = accuracy_score(y_test_classes, y_pred)
bal_acc = balanced_accuracy_score(y_test_classes, y_pred)
cm = confusion_matrix(y_test_classes, y_pred)
cr = classification_report(y_test_classes, y_pred)
print("Accuracy:", acc)
print("Balanced Accuracy:", bal_acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

# Plot ROC curve for each class
y_test_binarized = label_binarize(y_test_classes, classes=range(num_classes))

# Calculate ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### TRAINING SET ###

# Predictions
y_train_pred_prob = best_model.predict_proba(X_train)
y_train_pred = np.argmax(y_train_pred_prob, axis=1)
y_train_classes = np.argmax(y_train, axis=1)

# Calculate metrics
train_acc = accuracy_score(y_train_classes, y_train_pred)
train_bal_acc = balanced_accuracy_score(y_train_classes, y_train_pred)
train_cm = confusion_matrix(y_train_classes, y_train_pred)
train_cr = classification_report(y_train_classes, y_train_pred)
print("Training Set Accuracy:", train_acc)
print("Training Set Balanced Accuracy:", train_bal_acc)
print("Confusion Matrix on Training Data:\n", train_cm)
print("Classification Report on Training Data:\n", train_cr)
