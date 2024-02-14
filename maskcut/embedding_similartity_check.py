import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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


import numpy as np
import os
from scipy.spatial.distance import cosine
from collections import defaultdict

# Replace 'base_directory' with the path to your base directory where the folders are located
base_directory = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/"
output_file = "/home/jungseok/data/zpool_dataset/similarity_rankings.txt"

def load_features_and_filenames(base_dir):
    features_dict = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                features = np.load(file_path)
                # Store the features and the associated filename
                features_dict[file] = features
    return features_dict

def calculate_means(features_dict):
    mean_features_dict = {filename: np.mean(features, axis=1) for filename, features in features_dict.items()}
    return mean_features_dict

def rank_cosine_similarity(mean_features_dict):
    # Prepare a list to hold tuples of (filename1, filename2, similarity_score)
    similarity_scores = []
    
    filenames = list(mean_features_dict.keys())
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            filename1, filename2 = filenames[i], filenames[j]
            mean1, mean2 = mean_features_dict[filename1], mean_features_dict[filename2]
            # # Use loaded_features as needed
            feature1_torch = torch.from_numpy(mean1)
            feature2_torch = torch.from_numpy(mean2)

            # Assuming mean_feature1 and mean_feature2 are PyTorch tensors
            cosine_similarity = torch.nn.functional.cosine_similarity(feature1_torch.unsqueeze(0), feature2_torch.unsqueeze(0))
            similarity_scores.append((filename1, filename2, cosine_similarity))
    
    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[2], reverse=True)
    
    return similarity_scores

def display_ranked_pairs(similarity_scores):
    print("Ranking of pairs based on cosine similarity:")
    for rank, (filename1, filename2, similarity) in enumerate(similarity_scores, start=1):
        print(f"{rank}. {filename1} - {filename2}: {similarity}")

def write_rankings_to_file(similarity_scores, output_path):
    with open(output_path, 'w') as f:
        f.write("Ranking of pairs based on cosine similarity:\n")
        for rank, (filename1, filename2, similarity) in enumerate(similarity_scores, start=1):
            f.write(f"{rank}. {filename1} - {filename2}: {similarity}\n")

# Main execution flow
features_dict = load_features_and_filenames(base_directory)
mean_features_dict = calculate_means(features_dict)
similarity_scores = rank_cosine_similarity(mean_features_dict)
# display_ranked_pairs(similarity_scores)
write_rankings_to_file(similarity_scores, output_file)

print(f"Rankings have been saved to {output_file}")