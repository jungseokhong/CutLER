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

from CutLER.maskcut import dino # model ##
from CutLER.third_party.TokenCut.unsupervised_saliency_detection import metric ##
from CutLER.maskcut.crf import densecrf ##
from CutLER.maskcut.maskcut_v2 import maskcut, maskcut_img ##
# from CutLER.maskcut.maskcut_dinov2 import maskcut, maskcut_img ##

from CutLER.maskcut import img_save
from CutLER.maskcut import downsample

# import dino # model ##
# from third_party.TokenCut.unsupervised_saliency_detection import metric ##
# from crf import densecrf ##
# from maskcut_v2 import maskcut, maskcut_img ##
# from CutLER.maskcut.maskcut_dinov2 import maskcut, maskcut_img ##

# import img_save
# import downsample

from typing import List

from umap import UMAP
umap_reducer1 = UMAP(n_components=1, random_state=42)
umap_reducer3 = UMAP(n_components=3, random_state=42)


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

def maskcut_demo(extractor, imgs: List[Image.Image], backbone, patch_size, tau, N, fixed_size, cpu, output_path=None):

    for img in imgs:

        bipartitions, _, I_new, feat = maskcut_img(img, backbone, patch_size, tau, \
            N=N, fixed_size=fixed_size, cpu=cpu)

        I = img.convert('RGB')
        width, height = I.size
        pseudo_mask_list = []
        pseudo_mask_square_list = []
        latent_centroids = []
        latent_centroids_umap384_1 = []
        latent_centroids_umap384_3 = []
        down_pseudo_mask_list = []
        pos_centroids = []

        for idx, bipartition in enumerate(bipartitions):
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            if not cpu:
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
            # if np.any(pseudo_mask[0, :] == 1) or np.any(pseudo_mask[-1, :] == 1) or \
            # np.any(pseudo_mask[:, 0] == 1) or np.any(pseudo_mask[:, -1] == 1):
            #     print("Edge values are 1")
            #     continue  # Skip this mask if edge values are 1

            if np.any(pseudo_mask[:, 0] == 1) or np.any(pseudo_mask[:, -1] == 1):
                print("Edge values are 1")
                continue  # Skip this mask if edge values are 1

            # New code to filter out large pseudo_masks
            if np.sum(pseudo_mask) > 0.1 * np.size(pseudo_mask):
                print("Mask is larger than 10 percent of total pixels")
                continue  # Skip this mask if it's larger than 5% of total pixels

            # Calculate the percentage of the bottom edge taken by the mask
            bottom_row = pseudo_mask[-1, :]  # Get the bottom row
            bottom_edge_ones = np.sum(bottom_row == 1)  # Count pixels set to 1
            total_pixels_in_row = bottom_row.size  # Total pixels in the bottom row
            percentage_of_ones = (bottom_edge_ones / total_pixels_in_row) * 100  # Calculate the percentage

            # Check if more than 5% of the bottom edge is taken by the mask
            if percentage_of_ones > 5:
                print("More than 5% of the bottom edge is taken by the mask, skipping this mask.")
                continue  # Skip further processing of this mask


            # Calculate the number of rows to consider (30% of total rows)
            num_rows_to_consider = int(np.ceil(pseudo_mask.shape[0] * 0.3))

            # # Calculate the sum of the first 30% of rows
            # if np.sum(pseudo_mask[:num_rows_to_consider]) > 50:
            #     print("Sum of first 30% of rows is greater than 50")
            #     continue 

            # on the floor and sum is greater than 130...
            # maybe on the floor can be filtered by indices..


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
                extracted_features = feat[:, non_zero_indices]
                ## this is for the mean of the features
                mean_features = torch.mean(extracted_features, dim=1)
                latent_centroids.append(mean_features)

                ## this is for the UMAP of the features
                extracted_features = extracted_features.cpu().detach().numpy()
                umap_features1 = umap_reducer1.fit_transform(extracted_features)
                umap_features1 = torch.from_numpy(umap_features1.flatten())
                umap_features3 = umap_reducer3.fit_transform(extracted_features)
                umap_features3 = torch.from_numpy(umap_features3.flatten())
                # print(f'umap_features shape: {umap_features.shape}')
                latent_centroids_umap384_1.append(umap_features1)
                latent_centroids_umap384_3.append(umap_features3)

                # img_save.save_numpy_array_as_image(down_pseudo_mask, "square_mask"+str(id)+"_"+str(non_zero_indices.shape[0])+".jpg")

                centroid = find_centroid(resized_pseudo_mask)  # returns x, y
                x_percent = centroid[0] / resized_pseudo_mask.shape[1]  # X?
                y_percent = centroid[1] / resized_pseudo_mask.shape[0]  # Y?
                pos_centroids.append([x_percent, y_percent])
                # img_save.save_numpy_array_as_image(resized_pseudo_mask, "resized_mask"+str(id)+"x"+str(x_percent)+"y"+str(y_percent)+"_"+str(non_zero_indices.shape[0])+".jpg")
                pseudo_mask_bool = resized_pseudo_mask.astype(bool)
                binary_mask += pseudo_mask_bool

            # Process resized_pseudo_mask for visualization and centroid calculation

            # if len(np.nonzero(resized_pseudo_mask.flatten())[0]) > 1:
            #     centroid = find_centroid(resized_pseudo_mask)  # returns x, y
            #     x_percent = centroid[0] / resized_pseudo_mask.shape[1]  # X?
            #     y_percent = centroid[1] / resized_pseudo_mask.shape[0]  # Y?
            #     pos_centroids.append([x_percent, y_percent])
            #     img_save.save_numpy_array_as_image(resized_pseudo_mask, "resized_mask"+str(id)+".jpg")
            #     pseudo_mask_bool = resized_pseudo_mask.astype(bool)
            #     binary_mask += pseudo_mask_bool

            input = vis_mask(input, resized_pseudo_mask, random_color(rgb=True))
            if output_path != None:
                input.save(os.path.join(output_path, "demo.jpg"))

            id += 1        
        
        
        
        # id = 0
        # ## feat is torch.size([384,3600])
        # # print(f'feat.shape: {feat.shape} {feat}')
        # for pseudo_mask in pseudo_mask_square_list:
        #     print(f'pseudo_mask shape: {pseudo_mask.shape}')
        #     down_pseudo_mask =downsample.downsample_numpy_array(pseudo_mask, (60,60)) ######
        #     down_pseudo_mask_list.append(down_pseudo_mask)
        #     # print(f'down_pseudo_mask shape: {down_pseudo_mask.shape}')

        #     # Flatten the downsampled mask and find non-zero indices
        #     flat_mask = down_pseudo_mask.flatten()
        #     non_zero_indices = np.nonzero(flat_mask)[0]
        #     print(f'non_zero_indices shape: {non_zero_indices.shape}')

        #     # Extract features from feat using non-zero indices
        #     # Assuming feat is a tensor of shape [feature_dim, num_patches]
        #     if non_zero_indices.shape[0] > 0:
        #         extracted_features = feat[:, non_zero_indices]
        #         print(f'extracted_features shape: {extracted_features.shape}')

        #         # Computing the mean of extracted features
        #         # [TODO] potentially add clustering here
        #         mean_features = torch.mean(extracted_features, dim=1)
        #         latent_centroids.append(mean_features)
        #         # print(f'mean_features shape: {mean_features.shape}')

        #         img_save.save_numpy_array_as_image(down_pseudo_mask, "mask"+str(id)+".jpg")
        #     id = id + 1

        # input = np.array(I)
        # binary_mask = np.zeros((height, width), dtype=np.uint8)

        # for pseudo_mask in pseudo_mask_list:

        #     input = vis_mask(input, pseudo_mask, random_color(rgb=True))
        #     print(f'pseudo_mask shape before flatten: {pseudo_mask.shape}')
        #     if len(np.nonzero(pseudo_mask.flatten())[0])>1:
        #         centroid = find_centroid(pseudo_mask) # return x, y
        #         x_percent = centroid[0]/pseudo_mask.shape[1] ## X?
        #         y_percent = centroid[1]/pseudo_mask.shape[0] ## Y?
        #         pos_centroids.append([x_percent, y_percent])  
        #         print(f'pseudo_mask shape: {pseudo_mask.shape}')
        #         pseudo_mask_bool = pseudo_mask.astype(bool)
        #         binary_mask += pseudo_mask_bool
        #     if output_path != None:
        #         input.save(os.path.join(output_path, "demo.jpg"))

        img_save.save_numpy_array_as_image(binary_mask*255, "binary_mask.jpg")
        segmentation_masks = [binary_mask]

        # To guarantee that input is a PIL image
        input = Image.fromarray(np.uint8(input))

    return segmentation_masks, input, latent_centroids, latent_centroids_umap384_1, latent_centroids_umap384_3, pos_centroids 



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

    I = Image.open(args.img_path).convert('RGB')
    imgs = [I]
    segmentation_masks, input, latent_centroids, pos_centroids  =maskcut_demo(None, imgs, backbone, args.patch_size, 
                                                                    args.tau, args.N, args.fixed_size, args.cpu, output_path=None)


'''

    bipartitions, _, I_new, feat = maskcut(args.img_path, backbone, args.patch_size, args.tau, \
        N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)

    I = Image.open(args.img_path).convert('RGB')
    width, height = I.size
    pseudo_mask_list = []
    pseudo_mask_square_list = []
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
        pseudo_mask_list.append(pseudo_mask)

    id = 0
    ## feat is torch.size([384,3600])
    # print(f'feat.shape: {feat.shape} {feat}')
    for pseudo_mask in pseudo_mask_square_list:
        print(f'pseudo_mask shape: {pseudo_mask.shape}')
        down_pseudo_mask =downsample.downsample_numpy_array(pseudo_mask)
        print(f'down_pseudo_mask shape: {down_pseudo_mask.shape}')

        # Flatten the downsampled mask and find non-zero indices
        flat_mask = down_pseudo_mask.flatten()
        non_zero_indices = np.nonzero(flat_mask)[0]
        print(f'non_zero_indices shape: {non_zero_indices.shape}')

        # Extract features from feat using non-zero indices
        # Assuming feat is a tensor of shape [feature_dim, num_patches]
        if non_zero_indices.shape[0] > 0:
            extracted_features = feat[:, non_zero_indices]
            print(f'extracted_features shape: {extracted_features.shape}')

            # Computing the mean of extracted features
            mean_features = torch.mean(extracted_features, dim=1)
            print(f'mean_features shape: {mean_features.shape}')

        img_save.save_numpy_array_as_image(down_pseudo_mask, "mask"+str(id)+".jpg")
        id = id + 1

    input = np.array(I)
    for pseudo_mask in pseudo_mask_list:
        input = vis_mask(input, pseudo_mask, random_color(rgb=True))
    input.save(os.path.join(args.output_path, "demo.jpg"))

'''