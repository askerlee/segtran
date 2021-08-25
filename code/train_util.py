import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloaders.datasets2d
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import cv2
import imgaug.augmenters as iaa
import imgaug as ia
from torchvision import transforms

def init_augmentation(args):
    if args.randscale > 0:
        crop_percents = (-args.randscale, args.randscale)
    else:
        crop_percents = (0, 0)
    
    # Images after augmentation/transformation should keep their original size orig_input_size.  
    # Will be resized before fed into the model.  
    tgt_width, tgt_height = args.orig_input_size
    
    if args.do_affine:
        affine_prob = 0.3
    else:
        affine_prob = 0
        
    common_aug_func =   iaa.Sequential(
                            [
                                # resize the image to the shape of orig_input_size
                                iaa.Resize({'height': tgt_height, 'width': tgt_width}),   
                                iaa.Sometimes(0.5, iaa.CropAndPad(
                                    percent=crop_percents,
                                    pad_mode='constant', # ia.ALL,
                                    pad_cval=0
                                )),
                                # apply the following augmenters to most images
                                iaa.Fliplr(0.2),  # Horizontally flip 20% of all images
                                iaa.Flipud(0.2),  # Vertically flip 20% of all images
                                iaa.Sometimes(0.3, iaa.Rot90((1,3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                                # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                                iaa.Sometimes(affine_prob, iaa.Affine(
                                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                                        shear=(-16, 16), # shear by -16 to +16 degrees
                                        order=1,
                                        cval=(0,255),
                                        mode='reflect'
                                )),
                                # iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades.
                                # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                                # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                                iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),    
                                iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                                iaa.Grayscale(alpha=args.gray_alpha)
                            ])
                            
    image_aug_func = transforms.RandomChoice([
                        transforms.ColorJitter(brightness=0.2),
                        transforms.ColorJitter(contrast=0.2), 
                        transforms.ColorJitter(saturation=0.2),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0), 
                     ])
                             
    if args.robust_aug_types is None:
        # robust_aug_func does nothing.
        robust_aug_funcs = [ transforms.Pad(0) ]
    else:
        robust_aug_funcs = []
        for aug_type in args.robust_aug_types:
            if aug_type == 'brightness':
                robust_aug_func = transforms.ColorJitter(brightness=args.robust_aug_degrees)
            elif aug_type == 'contrast':
                robust_aug_func = transforms.ColorJitter(contrast=args.robust_aug_degrees)
            else:
                breakpoint()
            robust_aug_funcs.append(robust_aug_func)
        print("Robustness augmentation: {}".format(robust_aug_funcs))
    
    return common_aug_func, image_aug_func, robust_aug_funcs
    
def init_training_dataset(args, ds_settings, ds_name, ds_split, train_data_path, sample_num,
                          common_aug_func, image_aug_func, robust_aug_funcs):
    DataSetClass = dataloaders.datasets2d.__dict__[args.ds_class]    
    ds_weight    = ds_settings['weight'][ds_name]
    if 'uncropped_size' in ds_settings:
        uncropped_size  = ds_settings['uncropped_size'][ds_name]
    else:
        uncropped_size  = -1

    if uncropped_size == -1 and 'orig_dir' in ds_settings:
        orig_dir  = ds_settings['orig_dir'][ds_name]
        orig_ext  = ds_settings['orig_ext'][ds_name]
    else:
        orig_dir = orig_ext = None
                    
    has_mask      = ds_settings['has_mask'][ds_name]
    mean          = ds_settings['mean'][ds_name]
    std           = ds_settings['std'][ds_name]
    image_trans_func = transforms.Compose(
                            robust_aug_funcs + [   
                            image_aug_func,
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std) ]
                       )
                    
    db_train = DataSetClass(base_dir=train_data_path,
                            split=ds_split,
                            mode='train',
                            sample_num=sample_num,
                            mask_num_classes=args.num_classes,
                            has_mask=has_mask,
                            ds_weight=ds_weight,
                            common_aug_func=common_aug_func,
                            image_trans_func=image_trans_func,
                            segmap_trans_func=None,
                            binarize=args.binarize,
                            train_loc_prob=args.localization_prob,
                            chosen_size=args.orig_input_size[0],   # ignored in SegWhole instances.
                            uncropped_size=uncropped_size,
                            min_output_size=args.patch_size,
                            orig_dir=orig_dir,
                            orig_ext=orig_ext)
                            
    print("{}: {} images, uncropped_size: {}, has_mask: {}".format(
            ds_name, len(db_train), uncropped_size, has_mask))
    return db_train

# Replace BatchNorm with GroupNorm
# https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686
def bn2gn(model, old_layer_type, new_layer_type, num_groups, convert_weights):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = bn2gn(module, old_layer_type, new_layer_type, num_groups, convert_weights)

        # single module
        if type(module) == old_layer_type:
            old_layer = module
            new_layer = new_layer_type(num_groups, module.num_features, module.eps, module.affine) 
            if convert_weights:
                new_layer.weight = old_layer.weight
                new_layer.bias = old_layer.bias

            model._modules[name] = new_layer

    return model

def remove_norms(model, name):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = remove_norms(module, name)

        if type(module) == nn.LayerNorm or \
           type(module) == nn.BatchNorm2d or \
           type(module) == nn.GroupNorm:
            do_nothing = nn.Identity()
            model._modules[name] = do_nothing

    return model

def batch_norm(t4d, debug=False):
    chan_num = t4d.shape[1]
    t4d_chanfirst = t4d.transpose(0, 1)
    t4d_flat = t4d_chanfirst.reshape(chan_num, -1)
    stds  = t4d_flat.std(dim=1)
    means = t4d_flat.mean(dim=1)
    t4d_normed = (t4d_flat - means.view(chan_num, 1)) / stds.view(chan_num, 1)
    t4d_normed = t4d_normed.reshape(t4d_chanfirst.shape).transpose(0, 1)
    return t4d_normed
    
def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.training = False

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 and m.affine:
        m.weight.requires_grad = False
        m.bias.requires_grad = False
            
def reset_parameters(module):
    reset_count = 0
    for layer in module.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            reset_count += 1
    # print("%d layers reset" %reset_count)
