import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from umap import UMAP

# features_path1 = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/image_1705883246926492239/image_1705883246926492239_x0.471875_y0.8425925925925926_features.npy"
# features_path2 = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/image_1705883341228864998/image_1705883341228864998_x0.3489583333333333_y0.5509259259259259_features.npy"
# # Load the NumPy array
# loaded_features1 = np.load(features_path1)
# loaded_features2 = np.load(features_path2)
# print(f"loaded_features1.shape: {loaded_features1.shape}")
# print(f"loaded_features2.shape: {loaded_features2.shape}")
# mean_features1 = np.mean(loaded_features1, axis=1)
# mean_features2 = np.mean(loaded_features2, axis=1)
# print(f"mean_features1.shape: {mean_features1.shape}")
# print(f"mean_features2.shape: {mean_features2.shape}")
# # # Use loaded_features as needed
# feature1_torch = torch.from_numpy(mean_features1)
# feature2_torch = torch.from_numpy(mean_features2)

# # Assuming mean_feature1 and mean_feature2 are PyTorch tensors
# cosine_similarity = torch.nn.functional.cosine_similarity(feature1_torch.unsqueeze(0), feature2_torch.unsqueeze(0))

# print(f"Cosine Similarity: {cosine_similarity.item()}")

# # Initialize UMAP. n_components is set to 1 since we want to reduce the data to a single dimension
# umap_reducer = UMAP(n_components=1, random_state=42)

# # Fit and transform the matrices using UMAP
# reduced_mat1 = umap_reducer.fit_transform(loaded_features1)
# reduced_mat2 = umap_reducer.fit_transform(loaded_features2)

# # reduced_mat1 and reduced_mat2 now have shapes (384, 1). 
# # If you specifically want them in the shape (384,), you can simply reshape or flatten the arrays
# reduced_mat1 = torch.from_numpy(reduced_mat1.flatten())
# reduced_mat2 = torch.from_numpy(reduced_mat2.flatten())
# umap_cosine_similarity = torch.nn.functional.cosine_similarity(reduced_mat1.unsqueeze(0), reduced_mat2.unsqueeze(0))
# print(f"Cosine Similarity UMAP: {umap_cosine_similarity.item()}")

import numpy as np
import os
from scipy.spatial.distance import cosine
from collections import defaultdict

# Replace 'base_directory' with the path to your base directory where the folders are located
base_directory = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/"
output_file = "/home/jungseok/data/zpool_dataset/umap_similarity_rankings_384_1.txt"


# umap_reducer = UMAP(n_components=1)
# umap_reducer = UMAP(n_components=1, random_state=42)


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

def reduce_dimensions_with_umap(features_dict):
    umap_reducer = UMAP(n_components=1, random_state=42)
    reduced_features_dict = {}
    for filename, features in features_dict.items():
        # Reduce dimensions of the loaded features using UMAP
        reduced_features = umap_reducer.fit_transform(features).flatten()
        reduced_features_dict[filename] = reduced_features
    return reduced_features_dict

# def calculate_umap(features_dict):

#     umap_features_dict = {filename: umap_reducer.fit_transform(features) for filename, features in features_dict.items()}
#     return umap_features_dict

def rank_cosine_similarity(reduced_features_dict):
    # Prepare a list to hold tuples of (filename1, filename2, similarity_score)
    similarity_scores = []
    
    filenames = list(reduced_features_dict.keys())
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            filename1, filename2 = filenames[i], filenames[j]
            reduced1, reduced2 = reduced_features_dict[filename1], reduced_features_dict[filename2]
            # Convert reduced features to PyTorch tensors
            feature1_torch = torch.from_numpy(reduced1)
            feature2_torch = torch.from_numpy(reduced2)
            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(feature1_torch.unsqueeze(0), feature2_torch.unsqueeze(0))
            similarity_scores.append((filename1, filename2, cosine_similarity.item()))
    
    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[2], reverse=True)
    
    return similarity_scores

def display_ranked_pairs(similarity_scores):
    print("Ranking of pairs based on cosine similarity:")
    for rank, (filename1, filename2, similarity) in enumerate(similarity_scores, start=1):
        print(f"{rank}. {filename1} - {filename2}: {similarity}")

def write_rankings_to_file(similarity_scores, output_path):
    with open(output_path, 'w') as f:
        # f.write("Ranking of pairs based on cosine similarity:\n")
        for rank, (filename1, filename2, similarity) in enumerate(similarity_scores, start=1):
            f.write(f"{rank}. {filename1} - {filename2}: {similarity}\n")

# Main execution flow
features_dict = load_features_and_filenames(base_directory)
reduced_features_dict = reduce_dimensions_with_umap(features_dict)
similarity_scores = rank_cosine_similarity(reduced_features_dict)
write_rankings_to_file(similarity_scores, output_file)
# display_ranked_pairs(similarity_scores)

print(f"Rankings have been saved to {output_file}")