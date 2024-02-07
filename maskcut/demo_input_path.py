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
