import cv2
import pandas as pd
import numpy as np
from scipy.stats import entropy
import os

# === CONFIGURATION ===

# Replace this with your local dataset path
base_path = "/path/to/Bone Renders/"

# List of relative folder names and their binary labels (0 = tibia, 1 = radius)
train_folders = [
    "0001", "0021", "0029", "0051", "0053",
    "0056", "0067", "0068", "0102", "0105",
    "Box 1 #2 - 1", "Box 1 #3 - 1", "Box 2 #1 - 0",
    "Box 2 #2 - 0", "Box 3 #1 - 0", "Box 3 #2 - 1",
    "Box 3 #3 - 1", "Box 4 #1 - 0", "Box 4 #2 - 0"
]

labels = [
    '0', '1', '0', '0', '1', '0', '1', '1', '1', '0',
    '1', '1', '0', '0', '0', '1', '1', '0', '0'
]

# === FEATURE EXTRACTION ===

flattened_features_per_image = []
target_column = []

for folder_idx, folder_name in enumerate(train_folders):
    folder_path = os.path.join(base_path, folder_name)

    for image_number in range(100):  # view_000.png to view_099.png
        image_filename = f"view_{image_number:03d}.png"
        image_path = os.path.join(folder_path, image_filename)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img_gray, (300, 300))
        img_flat = resized_img.reshape(-1)

        image_features = []

        for theta in range(4):
            theta_val = theta / 4.0 * np.pi
            for sigma in (1, 3, 5):
                for lamda in (0, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.5):
                        ksize = 20
                        phi = 0
                        kernel = cv2.getGaborKernel(
                            (ksize, ksize), sigma, theta_val, lamda, gamma, phi, ktype=cv2.CV_32F
                        )
                        fimg = cv2.filter2D(img_flat, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)

                        mean = np.mean(filtered_img)
                        std = np.std(filtered_img)
                        energy = np.sum(filtered_img**2)
                        hist, _ = np.histogram(filtered_img, bins=256, range=(0, 255), density=True)
                        entropy_value = entropy(hist)
                        contrast = np.sum((filtered_img - np.mean(filtered_img))**2)
                        homogeneity = np.sum(filtered_img / (1 + np.abs(filtered_img - np.mean(filtered_img))))

                        image_features.extend([
                            mean, std, energy, entropy_value, contrast, homogeneity
                        ])

        flattened_features_per_image.append(image_features)
        target_column.append(labels[folder_idx])

# === SAVE OUTPUT ===

# Create column names
gabor_columns = []
for i in range(1, 73):  # 4x3x3x2 = 72 Gabor kernels
    gabor_columns.extend([
        f'gabor{i}_mean', f'gabor{i}_std', f'gabor{i}_energy',
        f'gabor{i}_entropy', f'gabor{i}_contrast', f'gabor{i}_homogeneity'
    ])

# Build DataFrame
final_df = pd.DataFrame(flattened_features_per_image, columns=gabor_columns)
final_df['target'] = target_column

# Save to CSV
final_df.to_csv('Gabor_Feature_Dataset.csv', index=False)
print("✅ Feature extraction complete. Saved to 'Gabor_Feature_Dataset.csv'")
