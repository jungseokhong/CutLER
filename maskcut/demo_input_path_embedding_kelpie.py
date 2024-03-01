#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
from detectron2.utils.colormap import random_color

import dino # model
from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut_v2 import maskcut, maskcut_img

import img_save
import downsample

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)

def find_centroid(pseudo_mask):
    """
    Find the centroid of a binary mask.

    Args:
    pseudo_mask (numpy.ndarray): A 2D numpy array where non-zero values represent the mask.

    Returns:
    tuple: The (x, y) coordinates of the centroid.
    """
    # Ensure pseudo_mask is a binary mask
    pseudo_mask = np.where(pseudo_mask > 0, 1, 0)

    # Calculate the moments of the binary image
    M = ndimage.measurements.center_of_mass(pseudo_mask)

    # The centroid is the center of mass
    centroid = (int(M[1]), int(M[0]))  # (x, y) format
    return centroid

def check_edges_for_ones(pseudo_mask):
    """
    Check if there is any row in the pseudo_mask that has 1s on both
    the left-most and right-most columns.

    Args:
    pseudo_mask (numpy.ndarray): A 2D numpy array where non-zero values
                                 represent the mask.

    Returns:
    bool: True if there is at least one row with 1s on both edges, else False.
    """
    # Iterate through each row in the pseudo_mask
    for row in pseudo_mask:
        if row[0] == 1 and row[-1] == 1:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut Demo')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')
    parser.add_argument('--tau', type=float, default=0.15, help='threshold used for producing binary graph')

    # additional arguments
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--output_path', type=str,  default='', help='path to save outputs')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with images')


    args = parser.parse_args()
    print (args)

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    # Iterate over each image in the input directory
    for filename in os.listdir(args.input_dir):
        img_path = os.path.join(args.input_dir, filename)
        if os.path.isfile(img_path):
            
            base_filename = os.path.splitext(filename)[0]  # Get the base name of the file
            image_specific_dir = os.path.join(args.out_dir, base_filename)  # Create a path for the directory


            bipartitions, _, I_new, feat = maskcut(img_path, backbone, args.patch_size, args.tau, \
                N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)

            I = Image.open(img_path).convert('RGB')
            width, height = I.size
            pseudo_mask_list = []
            pseudo_mask_square_list = []
            latent_centroids = []
            down_pseudo_mask_list = []
            pos_centroids = []



            for idx, bipartition in enumerate(bipartitions):
                # post-process pesudo-masks with CRF
                pseudo_mask = densecrf(np.array(I_new), bipartition)
                pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

                # filter out the mask that have a very different pseudo-mask after the CRF
                if not args.cpu:
                    mask1 = torch.from_numpy(bipartition).cuda()
                    mask2 = torch.from_numpy(pseudo_mask).cuda()
                else:
                    mask1 = torch.from_numpy(bipartition)
                    mask2 = torch.from_numpy(pseudo_mask)
                if metric.IoU(mask1, mask2) < 0.5:
                    pseudo_mask = pseudo_mask * -1

                # construct binary pseudo-masks
                pseudo_mask[pseudo_mask < 0] = 0

                # Heuristic filtering

                # # Check if edge values of pseudo_mask are 1
                # if np.any(pseudo_mask[0, :] == 1) and np.any(pseudo_mask[-1, :] == 1):
                #     print("side Edge values are 1")
                #     # continue  # Skip this mask if edge values are 1
                #     pass

                if check_edges_for_ones(pseudo_mask):
                    print("Found a row with 1s on both left-most and right-most columns.")
                    continue

                # New code to filter out large pseudo_masks
                # if np.sum(pseudo_mask) > 0.05 * np.size(pseudo_mask):
                    # print("Mask is larger than 5 percent of total pixels")
                    # continue  # Skip this mask if it's larger than 5% of total pixels
                    # pass
                
                if np.sum(pseudo_mask) > 0.4 * 480*480:
                    print("Mask is larger than 40 percent of total pixels")
                    continue  # Skip this mask if it's larger than 5% of total pixels


                # Calculate the number of rows to consider (30% of total rows)
                num_rows_to_consider = int(np.ceil(pseudo_mask.shape[0] * 0.3))

                # # Calculate the sum of the first 30% of rows
                # if np.sum(pseudo_mask[:num_rows_to_consider]) > 50:
                #     print("Sum of first 30% of rows is greater than 50")
                #     # continue
                #     pass 


                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
                # pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))
                pseudo_mask = np.asarray(pseudo_mask) ##

                pseudo_mask = pseudo_mask.astype(np.uint8)
                upper = np.max(pseudo_mask)
                lower = np.min(pseudo_mask)
                thresh = upper / 2.0
                pseudo_mask[pseudo_mask > thresh] = upper
                pseudo_mask[pseudo_mask <= thresh] = lower
                pseudo_mask_square_list.append(pseudo_mask) ##

                pseudo_mask = np.asarray(Image.fromarray(pseudo_mask).resize((width, height))) ##
                print(f'pseudo_mask shape: {pseudo_mask.shape} before list')

                pseudo_mask_list.append(pseudo_mask)

            


            combined_masks = zip(pseudo_mask_square_list, pseudo_mask_list)  # Combine the lists for parallel processing
            input = np.array(I)
            binary_mask = np.zeros((height, width), dtype=np.uint8)

            id = 0  # Reset id for each image in imgs
            for square_pseudo_mask, resized_pseudo_mask in combined_masks:
                # Process square_pseudo_mask for downsampling and feature extraction
                print(f'square_pseudo_mask shape: {square_pseudo_mask.shape}')
                print(f'resized_pseudo_mask shape: {resized_pseudo_mask.shape}')
                down_pseudo_mask = downsample.downsample_numpy_array(square_pseudo_mask, (60,60))
                down_pseudo_mask_list.append(down_pseudo_mask)
                flat_mask = down_pseudo_mask.flatten()
                non_zero_indices = np.nonzero(flat_mask)[0]
                if non_zero_indices.shape[0] > 0:
                    # create folder for each image using filename

                    os.makedirs(image_specific_dir, exist_ok=True)  # Create the directory


                    extracted_features = feat[:, non_zero_indices]
                    # Convert PyTorch tensor to NumPy array (if it's not already a NumPy array)
                    extracted_features_np = extracted_features.cpu().detach().numpy()

                    # # Define the path for saving features. This could be within the image-specific directory.
                    # features_path = os.path.join(image_specific_dir, f"{base_filename}_features.npy")

                    # # Save the NumPy array
                    # np.save(features_path, extracted_features_np)

                    mean_features = torch.mean(extracted_features, dim=1)
                    latent_centroids.append(mean_features)
                    down_filename = f"{base_filename}_square_mask{id}_{non_zero_indices.shape[0]}.jpg"
                    down_path = os.path.join(image_specific_dir, down_filename)
                    img_save.save_numpy_array_as_image(down_pseudo_mask, down_path)

                    centroid = find_centroid(resized_pseudo_mask)  # returns x, y
                    x_percent = centroid[0] / resized_pseudo_mask.shape[1]  # X?
                    y_percent = centroid[1] / resized_pseudo_mask.shape[0]  # Y?
                    pos_centroids.append([x_percent, y_percent])
                    resize_filename = f"{base_filename}_resized_mask{id}_x{x_percent}_y{y_percent}_{non_zero_indices.shape[0]}.jpg"
                    resize_path = os.path.join(image_specific_dir, resize_filename)
                    img_save.save_numpy_array_as_image(resized_pseudo_mask, resize_path)
                    pseudo_mask_bool = resized_pseudo_mask.astype(bool)
                    binary_mask += pseudo_mask_bool

                    # Define the path for saving features. This could be within the image-specific directory.
                    features_path = os.path.join(image_specific_dir, f"{base_filename}_x{x_percent}_y{y_percent}_features.npy")
                    # Save the NumPy array
                    np.save(features_path, extracted_features_np)


                id += 1        
            
            if np.sum(binary_mask) > 0:
                binary_path = os.path.join(image_specific_dir, f"{base_filename}_binary_mask.jpg")
                # if non_zero_indices.shape[0] > 0:
                img_save.save_numpy_array_as_image(binary_mask*255, binary_path)
                segmentation_masks = [binary_mask]

                # To guarantee that input is a PIL image
                input = Image.fromarray(np.uint8(input))
                for pseudo_mask in pseudo_mask_list:
                    input = vis_mask(input, pseudo_mask, random_color(rgb=True))
                os.makedirs(args.out_dir, exist_ok=True)
                output_filename = f"{base_filename}_seg.jpg"
                output_path = os.path.join(args.out_dir, output_filename)
                output_path2 = os.path.join(image_specific_dir, output_filename)
                input.save(output_path)  # Assuming `input` is your final processed image
                # if non_zero_indices.shape[0] > 0:
                input.save(output_path2)
                print(f'Saved output to {output_path}')
            else:
                input = Image.fromarray(np.uint8(input))
                os.makedirs(args.out_dir, exist_ok=True)
                output_filename = f"nomask_{filename}"
                output_path = os.path.join(args.out_dir, output_filename)
                input.save(output_path)
            # segmentation_masks, input, latent_centroids, pos_centroids 
                


# # Define the path to the saved features
# features_path = os.path.join(image_specific_dir, f"{base_filename}_features.npy")

# # Load the NumPy array
# loaded_features = np.load(features_path)

# # Use loaded_features as needed