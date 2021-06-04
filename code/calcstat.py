"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
import sys
import os
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import argparse
import imgaug.augmenters as iaa
import json
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--parent', dest='parent_dir', type=str, required=True, 
                    help='Parent folder of Image dataset(s).')
parser.add_argument('--dirs', dest='img_dirs', type=str, required=True, 
                    help='Root of Image dataset(s). Can specify multiple directories (separated with ",")')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5, 
                    help='Convert images to grayscale by so much degree.')
parser.add_argument('--size', dest='chosen_size', type=int, default=0, 
                    help='Use images of this size (among all cropping sizes). Default: 0, i.e., use all sizes.')
parser.add_argument('--chan', dest='channel_num', type=int, default=3, 
                    help='Number of image channels.')
parser.add_argument('--do', dest='command', type=str, default='stats', 
                    help='Command to execute.')
                    
args = parser.parse_args()
                    
# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = args.channel_num
img_types = ['png', 'jpg']

def cal_dir_stat(root, gray_alpha, chosen_size):
    cls_dirs = [ d for d in listdir(root) if isdir(join(root, d)) and 'masks' not in d ]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    gray_trans = iaa.Grayscale(alpha=gray_alpha)
    
    for idx, d in enumerate(cls_dirs):
        print("Class '{}'".format(d))
        im_paths = []
        for img_type in img_types:
            im_paths += glob(join(root, d, "*.{}".format(img_type)))
        if chosen_size:
            all_sizes_count = len(im_paths)
            im_paths = filter(lambda name: '_{}_'.format(chosen_size) in name, im_paths)
            im_paths = list(im_paths)
            print("{} size {} images chosen from {}".format(len(im_paths), chosen_size, all_sizes_count))
            
        for path in im_paths:
            im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channels in BGR order
            im = im[:, :, ::-1]   # Change channels to RGB
            im = gray_trans.augment_image(im)
            im = im/255.0
            pixel_num += (im.size/CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
        
        print("{} images counted".format(len(im_paths)))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))
    return rgb_mean, rgb_std

def check_masks(root, chosen_size):
    cls_dirs = [ d for d in listdir(root) if isdir(join(root, d)) and 'image' not in d ]
    whitelist = [0, 255]
    
    for idx, d in enumerate(cls_dirs):
        print("Class '{}'".format(d))
        im_paths = []
        for img_type in img_types:
            im_paths += glob(join(root, d, "*.{}".format(img_type)))
        if chosen_size:
            all_sizes_count = len(im_paths)
            im_paths = filter(lambda name: '_{}_'.format(chosen_size) in name, im_paths)
            im_paths = list(im_paths)
            print("{} size {} images chosen from {}".format(len(im_paths), chosen_size, all_sizes_count))
            
        for path in im_paths:
            im = cv2.imread(path)[:, :, 0]
            illegal = np.ones(im.shape[:2])
            for v in whitelist:
                illegal *= (im != v)
            if illegal.sum() > 0:
                pdb.set_trace()
    
# The script assumes that under img_dir_path, there are separate directories for each class
# of training images.
img_dirs = args.img_dirs.split(",")

if args.command == 'checkmask':
    for img_dir in img_dirs:
        img_dir_path = os.path.join(args.parent_dir, img_dir)
        print("Checking {}...".format(img_dir_path))
        check_masks(img_dir_path, args.chosen_size)
    exit(0)
    
ds2mean = {}
ds2std  = {}

for img_dir in img_dirs:
    img_dir_path = os.path.join(args.parent_dir, img_dir)
    print("Calculating {}...".format(img_dir_path))
    start = timeit.default_timer()
    mean, std = cal_dir_stat(img_dir_path, args.gray_alpha, args.chosen_size)
    end = timeit.default_timer()
    print("elapsed time: {:.1f}".format(end-start))
    mean_str = ", ".join([ "%.3f" %x for x in mean ])
    std_str  = ", ".join([ "%.3f" %x for x in std ])
    ds2mean[img_dir] = "[{}]".format(mean_str)
    ds2std[img_dir]  = "[{}]".format(std_str)
    if img_dir.endswith('-train'):
        img_dir2 = img_dir[:-6]
        ds2mean[img_dir2] = "[{}]".format(mean_str)
        ds2std[img_dir2]  = "[{}]".format(std_str)
            
    print("mean:\n[{}]\nstd:\n[{}]".format(mean_str, std_str))
    
json_dict = { 'mean': ds2mean, 'std': ds2std }
json_str = json.dumps(json_dict, indent=4, separators=(',', ': '))
json_str = json_str.replace('"[', '[')
json_str = json_str.replace(']"', ']')
print(json_str)
