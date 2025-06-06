import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Random Forest Classifier Example
# Load the dataset
# Make sure to replace '' with the path to your dataset
data = pd.read_csv('')

# Assuming the dataset has a 'target' column for classification
X = data.drop(columns=['target'])
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = rfc.score(X_test, y_test)
accuracy_percent = accuracy * 100

# Perform cross-validation
cv = cross_val_score(rfc, X, y, cv=5)
cv_mean = cv.mean() * 100

# Print the results
print(f'Test Set Accuracy: {accuracy_percent:.2f}%')
print(f'Cross Validation Accuracy: {cv_mean:.2f}%')