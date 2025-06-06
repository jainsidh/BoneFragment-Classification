import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the data
# Replace with your dataset path
data = pd.read_csv("")

X = data.drop(columns=["target"])
y = data["target"]

# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Grid Search for best 'k'
param_grid = {'n_neighbors': list(range(1, 21))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_knn = grid_search.best_estimator_
best_k = grid_search.best_params_['n_neighbors']
test_accuracy = best_knn.score(X_test, y_test)

# Print results
print(f"KNN Accuracy: {grid_search.best_score_ * 100:.2f}%")
