# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import argparse

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='./tmp',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--blender_root', type=str, default='./tmp',
    help='path for blender')
args = parser.parse_args()

save_folder = args.save_folder
dataset_folder = args.dataset_folder
blender_root = args.blender_root

synset_list = [
    #'02958343',  # Car
    '03001627',  # Chair
    '04379243'   # Table
    # '03790512'  # Motorbike
]
scale_list = [
    0.7,
    0.7
]

for synset, obj_scale in zip(synset_list, scale_list):
    file_list = sorted(os.listdir(os.path.join(dataset_folder, synset)))
    for idx, file in enumerate(file_list):

        # check if distination directory exists and contains
        skip = True
        root = os.path.join(save_folder, 'img', file)
        if os.path.isdir(root):
            img_root = os.path.join(root, 'models')
            if os.path.isdir(img_root):
                if len(os.listdir(img_root)) != 25:
                    skip = False
            else: skip = False
        else:
            skip = False

        if skip: continue

        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 >> tmp.out' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'models', 'model_normalized.obj'), obj_scale
        )
        os.system(render_cmd)
