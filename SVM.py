import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
# Replace the empty string with the path to your dataset
data = pd.read_csv("")

# Ensure the dataset has a 'target' column for classification
X_og = data.drop(columns=["target"])
y = data["target"]

# Scale the features to the range [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_og)

# Split the dataset into training and testing sets
# Adjust the test_size as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train SVM models with different kernels
# Linear and Polynomial kernels with different degrees
svm_model_1 = SVC(kernel='linear', C=250)
svm_model_2 = SVC(kernel='poly', degree = 2, C=250)
svm_model_3 = SVC(kernel='poly', degree = 3, C=250)
svm_model_4 = SVC(kernel='poly', degree = 4, C=250)
svm_model_5 = SVC(kernel='poly', degree = 5, C=250)

# Fit the models
svm_model_1.fit(X_train, y_train)
svm_model_2.fit(X_train, y_train)
svm_model_3.fit(X_train, y_train)
svm_model_4.fit(X_train, y_train)
svm_model_5.fit(X_train, y_train)

# Cross Validation
cv_scores_1 = cross_val_score(svm_model_1, X_train, y_train, cv=5)
cv_scores_2 = cross_val_score(svm_model_2, X_train, y_train, cv=5)
cv_scores_3 = cross_val_score(svm_model_3, X_train, y_train, cv=5)
cv_scores_4 = cross_val_score(svm_model_4, X_train, y_train, cv=5)
cv_scores_5 = cross_val_score(svm_model_5, X_train, y_train, cv=5)

# Mean cross-validation scores
mean_cv_score_1 = cv_scores_1.mean()
mean_cv_score_2 = cv_scores_2.mean()
mean_cv_score_3 = cv_scores_3.mean()
mean_cv_score_4 = cv_scores_4.mean()
mean_cv_score_5 = cv_scores_5.mean()

# Print the mean cross-validation scores
print("Mean Cross Validation Score for Linear Kernel: ", mean_cv_score_1)
print("Mean Cross Validation Score for SVM Poly (deg=2): ", mean_cv_score_2)
print("Mean Cross Validation Score for SVM Poly (deg=3): ", mean_cv_score_3)
print("Mean Cross Validation Score for SVM Poly (deg=4): ", mean_cv_score_4)
print("Mean Cross Validation Score for SVM Poly (deg=5): ", mean_cv_score_5)

# Plotting the mean cross-validation scores
mean_scores = [
    mean_cv_score_1,
    mean_cv_score_2,
    mean_cv_score_3,
    mean_cv_score_4,
    mean_cv_score_5
]

labels = ['Linear', 'Poly (2)', 'Poly (3)', 'Poly (4)', 'Poly (5)']

# Create a bar plot for the mean cross-validation scores
plt.figure()
plt.bar(labels, mean_scores, color = 'skyblue') 
plt.title('Mean Cross-Validation Scores for SVM Kernels')
plt.xlabel('SVM Kernel Type')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
