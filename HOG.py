import pandas as pd
import cv2
import os
from tqdm import tqdm
from skimage.feature import hog

# === CONFIGURATION ===

# Replace this with your local base path to the folders (Sample0011 to Sample0036)
base_path = "/path/to/Grayscale/GoodImg/"

# CSV output path
output_csv = "hog_output.csv"

hog_features_list = []
label_list = []

# === FEATURE EXTRACTION LOOP ===

label = 1  # Start from class label 1

for i in range(11, 37):  # Sample0011 to Sample0036
    folder_name = f"Sample00{i}"
    data_folder = os.path.join(base_path, folder_name)

    if not os.path.exists(data_folder):
        print(f"⚠️ Folder not found: {data_folder}")
        continue

    image_file_names = os.listdir(data_folder)

    for filename in tqdm(image_file_names, desc=f"Processing {folder_name}"):
        image_path = os.path.join(data_folder, filename)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue

        img = cv2.resize(img, (64, 64))

        # Extract HOG features
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )

        hog_features_list.append(features)
        label_list.append(label)

    label += 1

# === SAVE OUTPUT ===

df = pd.DataFrame(hog_features_list)
df['label'] = label_list  # Add class labels

df.to_csv(output_csv, index=False)
print(f"✅ HOG feature CSV saved to: {output_csv}")
