import numpy as np
import os
import torch

# path = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out"
# # Define the path to the saved features
# img_path = "image_1705883338986150194"
# intermediate_path = os.path.join(path, img_path)

# features_path = os.path.join(intermediate_path, "")

features_path1 = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/image_1705883338986150194/image_1705883338986150194_x0.2755208333333333_y0.7620370370370371_features.npy"
features_path2 = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/image_1705883338986150194/image_1705883338986150194_x0.9057291666666667_y0.44907407407407407_features.npy"
# Load the NumPy array
loaded_features1 = np.load(features_path1)
loaded_features2 = np.load(features_path2)
print(f"loaded_features1.shape: {loaded_features1.shape}")
print(f"loaded_features2.shape: {loaded_features2.shape}")
mean_features1 = np.mean(loaded_features1, axis=1)
mean_features2 = np.mean(loaded_features2, axis=1)
print(f"mean_features1.shape: {mean_features1.shape}")
print(f"mean_features2.shape: {mean_features2.shape}")
# # Use loaded_features as needed
feature1_torch = torch.from_numpy(mean_features1)
feature2_torch = torch.from_numpy(mean_features2)

# Assuming mean_feature1 and mean_feature2 are PyTorch tensors
cosine_similarity = torch.nn.functional.cosine_similarity(feature1_torch.unsqueeze(0), feature2_torch.unsqueeze(0))

print(f"Cosine Similarity: {cosine_similarity.item()}")