# 🤖 Classical Machine Learning Models for Bone Fragment Classification

This branch contains implementations of traditional machine learning algorithms used to classify deer bone fragments (tibia vs. radius) based on features extracted from 2D-rendered images. The features were derived using preprocessing techniques such as Gabor filtering, Histogram of Oriented Gradients (HOG), and statistical feature vectors.

Developed as part of the VIPER IPA Lab project at Purdue University.

---

## 📦 Models Implemented

| Model               | Description                                      |
|--------------------|--------------------------------------------------|
| `SVM`              | Support Vector Machine with linear and polynomial kernels |
| `XGBoost`          | Gradient Boosted Trees for robust classification |
| `KNN`              | k-Nearest Neighbors using Euclidean distance     |
| `Random Forest`    | Ensemble of decision trees                       |
| `Naive Bayes`      | Probabilistic model with Gaussian assumption     |
| `Logistic Regression` | Binary linear classification                   |

---

## 🧪 Input Features

All models use features extracted from the `preprocessing` branch, such as:
- **Gabor + Feature Vectors** (`Gabor_Feature_Dataset.csv`)
- **HOG Features** (`hog_output.csv`)

Each CSV contains 100s of features per image and a `target` column denoting the bone type:
- `0` – Tibia  
- `1` – Radius

---

## 📂 Files Included

| Script | Description |
|--------|-------------|
| `SVM.py` | Support Vector Machine Implementation |
| `KNN.py` | K-Nearest Neighbors Implementation |
| `Random Forest.py` | Random Forest Implementation |
| `Logistic Regression.py` | Logistic Regression Implementation |
| `XGBoost.py` | XGBoost Implementation |
| `Naive Bayes.py` | Naive Bayes Implementation |

---
