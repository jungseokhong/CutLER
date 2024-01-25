import numpy as np

def downsample_numpy_array(array, target_size=(60, 60)):
    """
    Downsamples a NumPy array to a target size by averaging and thresholding.

    Args:
    array: A NumPy array of size (480, 480).
    target_size: The target size to downsample to, default is (60, 60).

    Returns:
    A NumPy array downsampled to the target size.
    """
    # Calculate the size of blocks to average
    block_size = (array.shape[0] // target_size[0], array.shape[1] // target_size[1])

    # Reshape and compute the mean across 8x8 blocks
    downsampled = array.reshape(target_size[0], block_size[0], target_size[1], block_size[1]).mean(axis=(1, 3))

    # Apply the threshold
    downsampled = (downsampled > 0.5).astype(float)

    return downsampled