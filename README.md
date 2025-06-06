# 🦴 Bone Fragment Preprocessing Pipeline

This branch contains all preprocessing scripts used to convert rendered 2D images of deer bone fragments (tibia and radius) into meaningful features for classification models. These preprocessing steps are essential for extracting texture, shape, and statistical information from bone surface patterns.

---

## 🔧 Features Extracted

- **Gabor Filter Features**  
  Captures texture and directional information. Each image is filtered with multiple Gabor kernels and features like mean, std, entropy, contrast, and homogeneity are extracted.

- **HOG (Histogram of Oriented Gradients)**  
  Captures edge and shape descriptors from each image, suitable for classical ML models.

- **Feature Vectors**  
  Derived from Gabor-filtered images, including:
  - Mean
  - Standard Deviation
  - Energy
  - Entropy
  - Contrast
  - Homogeneity

---

## 🗂 Scripts Included

| Script | Description |
|--------|-------------|
| `Gabor Filters + Feature Vectors.py` | Extracts 72 Gabor-based angular information followed by data compression using Feature Vectors |
| `HOG.py` | Extracts HOG features from grayscale images in structured folders |

---
