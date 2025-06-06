import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
# Replace with your dataset path
data = pd.read_csv("")

# Features and target
X = data.drop(columns=["target"])
y = data["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100
print(f"XGBoost Accuracy: {accuracy_percent:.2f}%")

# Print classification report and confusion matrix
print(f"XGBoost Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
