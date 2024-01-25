import torch
from PIL import Image
import numpy as np

def save_painting_as_image(painting, file_path):
    """
    Save the 'painting' tensor as an image file.

    Args:
    painting: A PyTorch tensor representing the image.
    file_path: The path to save the image, including the file name and extension.
    """
    # Check if the tensor is on GPU and move it to CPU
    if painting.is_cuda:
        painting = painting.cpu()

    # Ensure the tensor is 2D (height x width)
    if painting.dim() > 2:
        painting = painting.squeeze()  # Remove channels dimension if it exists

    # Convert the tensor to a numpy array and scale to 0-255
    painting_np = (painting.numpy() * 255).astype(np.uint8)

    # Convert the numpy array to a PIL image
    painting_image = Image.fromarray(painting_np, mode='L')  # 'L' mode for grayscale

    # Save the image
    painting_image.save(file_path)
    print(f"Image saved at {file_path}")


def save_numpy_array_as_image(array, file_path):
    """
    Save a numpy array as an image.

    Args:
    array: A numpy array to be saved as an image.
    file_path: The path where the image will be saved, including the filename and extension.
    """
    # Ensure the array is in the right format (uint8)
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)

    # Convert the numpy array to a PIL image
    image = Image.fromarray(array)

    # Save the image
    image.save(file_path)
    print(f"Image saved at {file_path}")