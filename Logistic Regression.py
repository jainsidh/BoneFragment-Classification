import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
# Replace with your dataset path
data = pd.read_csv("")

# Features and target
X = data.drop(columns=["target"])
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = log_reg.score(X_test, y_test)

# Print accuracy
accuracy_percent = accuracy * 100
print(f"Logistic Regression Accuracy: {accuracy_percent:.2f}%")

# Print accuracy as a decimal
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
