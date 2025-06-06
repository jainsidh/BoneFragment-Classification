# 🧠 Deep Learning Models for Bone Fragment Classification

This branch contains deep learning models built to classify bone fragments (tibia vs. radius) from 2D images. These models were trained on grayscale or RGB image data rendered from 3D scans of deer bones and serve as an advanced comparison to classical ML methods.

Developed as part of the VIPER IPA Lab at Purdue University.

---

## 📦 Files Included

| File                   | Description |
|------------------------|-------------|
| `Multi_Layer_Perceptron.ipynb` | Implements a simple MLP architecture trained on flattened image data or extracted features |
| `CNN.ipynb`            | Convolutional Neural Network trained on 2D image inputs with spatial structure preserved |

---

## 📊 Model Overview

- **MLP (Multi-Layer Perceptron)**  
  A fully connected feedforward network using flattened input data. Useful when training on Gabor, HOG, or other pre-extracted features.

- **CNN (Convolutional Neural Network)**  
  Learns spatial features directly from raw 2D images. Ideal for identifying localized bone textures and shapes without manual preprocessing.

---

## 🗂 Input Data Format

- MLP: Feature dataset (e.g., Gabor or HOG features) in `.csv` format with a `target` column.
- CNN: Images stored in folders:
