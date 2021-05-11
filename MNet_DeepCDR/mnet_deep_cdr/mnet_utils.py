# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    # img = np.array(Image.fromarray(img).resize((input_size, input_size)).convert(3))
    img = np.array(Image.fromarray(img, mode='RGB').resize((input_size, input_size)))
    return img


def train_loader(data_list, data_path, mask_path, input_size):
    while 1:
        for lineIdx, temp_txt in enumerate(data_list):
            train_img = np.asarray(image.load_img(os.path.join(data_path, temp_txt),
                                                  target_size=(input_size, input_size, 3))
                                   ).astype('float32')
            img_mask = np.asarray(
                image.load_img(os.path.join(mask_path, temp_txt),
                               target_size=(input_size, input_size, 3))
            ) / 255.0

            train_img = np.reshape(train_img, (1,) + train_img.shape)
            img_mask = np.reshape(img_mask, (1,) + img_mask.shape)
            yield ([train_img], [img_mask, img_mask, img_mask, img_mask, img_mask])


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = [region.area for region in regions]
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return binary_fill_holes(np.asarray(binary).astype(int))


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef2(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score = 0.5 * score0 + 0.5 * score1

    return score


def dice_coef_loss(y_true, y_pred):
    return -dice_coef2(y_true, y_pred)

# C_x, C_y: coordinates of the center of the cropped area.
def disc_crop(org_img, DiscROI_size, C_x, C_y):
    tmp_size = int(DiscROI_size / 2)
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
    err_coord = [0, DiscROI_size, 0, DiscROI_size]

    if crop_coord[0] < 0:
        crop_coord[0] = 0
        crop_coord[1] = DiscROI_size
        
    if crop_coord[2] < 0:
        crop_coord[2] = 0
        crop_coord[3] = DiscROI_size
        
    if crop_coord[1] > org_img.shape[0]:
        crop_coord[1] = org_img.shape[0]
        crop_coord[0] = crop_coord[1] - DiscROI_size
        
    if crop_coord[3] > org_img.shape[1]:
        crop_coord[3] = org_img.shape[1]
        crop_coord[2] = crop_coord[3] - DiscROI_size

    disc_region = org_img[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
    return disc_region, crop_coord
        
    '''
    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[
                                                                          crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3],
                                                                          ]
    return disc_region, err_coord, crop_coord
                                                                          
    '''
    


def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def files_with_ext(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    print(len(file_list))
    return file_list
