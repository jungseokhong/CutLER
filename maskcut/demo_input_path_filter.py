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
from maskcut_v2 import maskcut

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
            

            bipartitions, _, I_new, feat = maskcut(img_path, backbone, args.patch_size, args.tau, \
                N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)

            I = Image.open(img_path).convert('RGB')
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
                if np.sum(pseudo_mask) > 0.1 * 480*480:
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

                # Calculate the sum of the top 50% of the rows
                rows_to_consider = int(pseudo_mask.shape[0] * 0.5)
                top_half_sum = np.sum(pseudo_mask[:rows_to_consider])

                # Calculate the sum of the entire pseudo_mask
                total_half_sum = 480*480*0.5

                # Calculate the percentage of the top half sum relative to the total sum
                top_half_percentage = (top_half_sum / total_half_sum) * 100

                # Check if the top half sum is greater than 5% of the total sum
                if top_half_percentage > 5:
                    print(f'filter fail: top_half_percentage: {top_half_percentage} and {filename}')
                    print("The sum of the top 50% of the rows is greater than 5% of the sum of the entire pseudo_mask, skipping this mask.")
                    continue  # Skip further processing of this mask
                print(f'filter pass: top_half_percentage: {top_half_percentage} and {filename}')

                if np.sum(pseudo_mask[80:421,:])/(340*480)> 0.069:
                    print("Sum of the middle 80% of rows is greater than 6.9%")
                    continue


                # ####### temp
                # # Find the first row with non-zero values
                # first_r = True
                # last_r = True
                # for first_nonzero_row, row in enumerate(pseudo_mask):
                #     if np.any(row) and first_r:
                #         print(f'first_nonzero_row: {first_nonzero_row} {filename}')
                #         break
                # else:
                #     first_nonzero_row = None

                # # Find the last row with non-zero values
                # for last_nonzero_row, row in enumerate(reversed(pseudo_mask)):
                #     if np.any(row) and last_r:
                #         print(f'last_nonzero_row: {last_nonzero_row} {filename}')
                #         break
                # else:
                #     last_nonzero_row = None

                # # Since we reversed the array to find the last non-zero row,
                # # we need to subtract from the total number of rows to get the correct index
                # if last_nonzero_row is not None:
                #     last_nonzero_row = pseudo_mask.shape[0] - 1 - last_nonzero_row
                #     print(f' real last_nonzero_row: {last_nonzero_row} {filename}')
                # #########



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
                pseudo_mask_list.append(pseudo_mask)

            id = 0
            ## feat is torch.size([384,3600])
            print(f'feat.shape: {feat.shape} {feat}')
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

                # img_save.save_numpy_array_as_image(down_pseudo_mask, "mask"+str(id)+".jpg") ## can put this back in
                id = id + 1

            input = np.array(I)
            for pseudo_mask in pseudo_mask_list:
                input = vis_mask(input, pseudo_mask, random_color(rgb=True))
            # input.save(os.path.join(args.output_path, "demo.jpg"))

            os.makedirs(args.out_dir, exist_ok=True)
            output_filename = "output_" + filename
            output_path = os.path.join(args.out_dir, output_filename)
            input.save(output_path)  # Assuming `input` is your final processed image
            print(f'Saved output to {output_path}')
