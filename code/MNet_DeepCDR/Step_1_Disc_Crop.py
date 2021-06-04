# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path
from sys import modules

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
from mnet_deep_cdr import Model_DiscSeg as DiscModel
from mnet_deep_cdr.mnet_utils import BW_img, disc_crop, mk_dir, files_with_ext
import pdb

discROI_size_list = [560] #[400, 500, 600, 700, 800]
DiscSeg_size = 640

data_types = ['jpg', 'png']
parent_dir = '/data/shaohua/MNet_DeepCDR'

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights(path.join(parent_dir, 'deep_model', 'Model_DiscSeg_ORIGA.h5'))

refuge_data_label_img_paths   = [ #[ 'Training400/Glaucoma',     'Annotation-Training400/Disc_Cup_Masks/Glaucoma',      'train_crop' ],
                                  #[ 'Training400/Non-Glaucoma', 'Annotation-Training400/Disc_Cup_Masks/Non-Glaucoma',  'train_crop' ],
                                  #[ 'Refuge2-Validation',        None,                                                 'valid2_crop' ],
                                  #[ 'REFUGE-Validation400',     'REFUGE-Validation400-GT/Disc_Cup_Masks',              'valid_crop' ],
                                  #[ 'Test400',                  'REFUGE-Test-GT/Disc_Cup_Masks',                       'test_crop' ]
                                  [ 'test2', None, 'test2_crop' ]
                                ]

external_data_label_img_paths = [ [ 'drishiti/image',     'drishiti/mask',      'drishiti_crop' ],
                                  [ 'rim-one/image',      'rim-one/mask',       'rim_crop' ],
                                ]
                                
refuge_data_dir     = '/data/shaohua/refuge2'
external_data_dir   = '/data/shaohua/fundus-external'

job = 'refuge'
if job == 'external':
    data_label_img_paths    = external_data_label_img_paths
    data_dir                = external_data_dir
    auto_crop = False
else:
    data_label_img_paths    = refuge_data_label_img_paths
    data_dir                = refuge_data_dir
    auto_crop = True

# Original size of external images are 800*800. Resize them to 640*640 before cropping.
manual_resize = 640

for data_img_path, label_img_path, save_dir in data_label_img_paths:
    data_img_path  = path.abspath(path.join(data_dir, data_img_path))
    data_save_path = mk_dir(path.join(data_dir, save_dir, 'images'))
    if label_img_path:
        label_img_path  = path.abspath(path.join(data_dir, label_img_path))
        label_save_path = mk_dir(path.join(data_dir, save_dir, 'masks'))

    file_test_list = []
    for data_type in data_types:
        file_test_list += files_with_ext(data_img_path, data_type)
        
    print('Process {data_img_path}: {img_count} images, save to {save_dir}'.format(
                data_img_path=data_img_path, img_count=len(file_test_list), save_dir=save_dir))

    for lineIdx, temp_txt in enumerate(file_test_list):
        # load image using keras. Convert to float numbers in [0, 1].
        orig_img = np.asarray(image.load_img(path.join(data_img_path, temp_txt)))
            
        if label_img_path:
            # load groundtruth label
            label_img_fullpath = path.join(label_img_path, temp_txt[:-4] + '.bmp')
            if not path.exists(label_img_fullpath):
                label_img_fullpath = path.join(label_img_path, temp_txt[:-4] + '.png')
                    
            orig_label = np.asarray(image.load_img(label_img_fullpath))[:, :, 0]
            new_label  = np.zeros(np.shape(orig_label) + (3,), dtype=np.uint8)
            # original label = 255: background
            # original label = 128: optic disc (excluding optic cup)
            # original label = 0:   optic cup
            #   255: background                       => 0   in channel 0
            # < 200: optic disc (including optic cup) => 255 in channel 0
            # < 100: optic cup only                   => 255 in channel 1
            new_label[orig_label < 200, 0] = 255    
            new_label[orig_label < 100, 1] = 255    
        else:
            new_label = None
        
        # Disc region detection by U-Net
        if auto_crop:    
            temp_img = resize(orig_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
            temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
            with tf.device('/gpu:0'):
                disc_map = DiscSeg_model.predict([temp_img])

            disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

            regions = regionprops(label(disc_map))
            # C_x, C_y: coordinates of the center of the cropped area.
            C_x = int(regions[0].centroid[0] * orig_img.shape[0] / DiscSeg_size)
            C_y = int(regions[0].centroid[1] * orig_img.shape[1] / DiscSeg_size)
        else:
            temp_img = resize(orig_img, (manual_resize, manual_resize, 3)) * 255
            C_x = C_y = manual_resize // 2
            orig_img = temp_img
            if label_img_path:
                new_label = resize(new_label, (manual_resize, manual_resize, 3)) * 255
                
        for disc_idx, discROI_size in enumerate(discROI_size_list):
            disc_region, crop_coord = disc_crop(orig_img, discROI_size, C_x, C_y)
            '''
            Disc_flat = rotate(cv2.linearPolar(disc_region, (discROI_size / 2, discROI_size / 2), discROI_size / 2,
                                               cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
            Label_flat = rotate(cv2.linearPolar(label_region, (discROI_size / 2, discROI_size / 2), discROI_size / 2,
                                                cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
            '''
            disc_result = Image.fromarray(disc_region.astype(np.uint8))
            filename = '{}_{}_{},{}.png'.format(temp_txt[:-4], discROI_size, crop_coord[0], crop_coord[2])
            disc_result.save(path.join(data_save_path, filename))

            print('Img {idx}: {temp_txt} => {newname}'.format(idx=lineIdx + 1, 
                                temp_txt=temp_txt, newname=filename))
                
            if label_img_path:
                label_region, _ = disc_crop(new_label, discROI_size, C_x, C_y)
                label_result = Image.fromarray(label_region.astype(np.uint8))
                label_result.save(path.join(label_save_path, filename))

plt.imshow(disc_result)
plt.show()
