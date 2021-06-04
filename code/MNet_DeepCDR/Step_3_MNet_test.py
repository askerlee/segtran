# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path
from sys import modules
from time import time

import cv2
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

from mnet_deep_cdr import Model_DiscSeg as DiscModel, Model_MNet as MNetModel
from mnet_deep_cdr.mnet_utils import pro_process, BW_img, disc_crop, mk_dir, files_with_ext

DiscROI_size = 600
DiscSeg_size = 640
CDRSeg_size = 400

parent_dir = path.dirname(resource_filename(modules[__name__].__name__, '__init__.py'))

test_data_path = path.join(parent_dir, 'test_img')
data_save_path = mk_dir(path.join(parent_dir, 'test_img'))

file_test_list = files_with_ext(test_data_path, '.jpg')

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights(path.join(parent_dir, 'deep_model', 'Model_DiscSeg_ORIGA.h5'))

CDRSeg_model = MNetModel.DeepModel(size_set=CDRSeg_size)
CDRSeg_model.load_weights(path.join(parent_dir, 'deep_model', 'Model_MNet_REFUGE.h5'))

for lineIdx, temp_txt in enumerate(file_test_list):
    # load image
    org_img = np.asarray(image.load_img(path.join(test_data_path, temp_txt)))
    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])
    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    disc_region, err_xy, crop_xy = disc_crop(org_img, DiscROI_size, C_x, C_y)

    # Disc and Cup segmentation by M-Net
    run_start = time()
    Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2),
                                       DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS), -90)

    temp_img = pro_process(Disc_flat, CDRSeg_size)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [_, _, _, _, prob_10] = CDRSeg_model.predict(temp_img)
    run_end = time()

    # Extract mask
    prob_map = np.reshape(prob_10, (prob_10.shape[1], prob_10.shape[2], prob_10.shape[3]))
    disc_map = np.array(Image.fromarray(prob_map[:, :, 0]).resize((DiscROI_size, DiscROI_size)))
    cup_map = np.array(Image.fromarray(prob_map[:, :, 1]).resize((DiscROI_size, DiscROI_size)))
    disc_map[-round(DiscROI_size / 3):, :] = 0
    cup_map[-round(DiscROI_size / 2):, :] = 0
    De_disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                  DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    De_cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                 DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    De_disc_map = np.array(BW_img(De_disc_map, 0.5), dtype=int)
    De_cup_map = np.array(BW_img(De_cup_map, 0.5), dtype=int)

    print('Processing Img {idx}: {temp_txt}, running time: {running_time}'.format(
        idx=lineIdx + 1, temp_txt=temp_txt, running_time=run_end - run_start
    ))

    # Save raw mask
    ROI_result = np.array(BW_img(De_disc_map, 0.5), dtype=int) + np.array(BW_img(De_cup_map, 0.5), dtype=int)
    Img_result = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.int8)
    Img_result[crop_xy[0]:crop_xy[1], crop_xy[2]:crop_xy[3], ] = ROI_result[err_xy[0]:err_xy[1], err_xy[2]:err_xy[3], ]
    save_result = Image.fromarray((Img_result * 127).astype(np.uint8))
    save_result.save(path.join(data_save_path, temp_txt[:-4] + '.png'))
