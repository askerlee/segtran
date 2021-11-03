import os
from os import listdir
from os.path import isdir, join
import re
import torch
import numpy as np
from glob import glob
from PIL import Image
import cv2
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import json
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from common_util import get_filename
import warnings
import csv

# Always keep the class dim at the first dim (without batch) or the second dim (after batch).
def index_to_onehot(mask, num_classes):
    if type(mask) == torch.Tensor:
        # Mask has a 1-element dim as the channel dim. 
        # This method is different from how to convert numpy array below.
        # The following method is more efficient for pytorch tensor.
        if mask.ndim == 4 and mask.shape[1] == 1:
            onehot_shape = list(mask.shape)
            onehot_shape[1] = num_classes
            mask_onehot = torch.zeros(onehot_shape, device=mask.device)
            mask_onehot.scatter_(1, mask.long(), 1)
        else:
            onehot_shape = list(mask.shape) + [num_classes]
            mask_onehot = torch.zeros(onehot_shape, device=mask.device)
            mask_onehot.scatter_(-1, mask.unsqueeze(-1).long(), 1)
            if mask.ndim == 3:
                mask_onehot = mask_onehot.permute([0, 3, 1, 2])
            elif mask.ndim == 2:
                mask_onehot = mask_onehot.permute([2, 0, 1])
            else:
                breakpoint()
    else:
        # Mask has a 1-element dim as the channel dim. Remove it.
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        mask_onehot = np.eye(num_classes)[mask]
        if mask.ndim == 4:
            mask_onehot = mask_onehot.transpose([0, 3, 1, 2])
        else:
            mask_onehot = mask_onehot.transpose([2, 0, 1])
    return mask_onehot

# *_inv_map functions always take hard masks as input.
def onehot_inv_map(mask_onehot, colormap=None):
    # Has the batch dimension
    if len(mask_onehot.shape) == 4:
        if type(mask_onehot) == np.ndarray:
            mask = np.zeros_like(mask_onehot[:, 0],    dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_onehot[:, 0], dtype=torch.uint8)
        num_classes = mask_onehot.shape[1]
        mask = mask_onehot.argmax(dim=1)
        if colormap is not None:
            mask = torch.index_select(colormap, 0, mask.flatten()).view(list(mask.shape)+[3])
        else:
            repeat = [ 1 for i in range(mask.ndim+1) ]
            repeat[-1] = 3
            mask = mask.unsqueeze(-1).repeat(*repeat)
            
    # Single image, no batch dimension
    elif len(mask_onehot.shape) == 3:
        if type(mask_onehot) == np.ndarray:
            mask = np.zeros_like(   mask_onehot[0], dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_onehot[0], dtype=torch.uint8)
        num_classes = mask_onehot.shape[0]
        mask = mask_onehot.argmax(dim=0)
        if colormap is not None:
            mask = torch.index_select(colormap, 0, mask.flatten()).view(list(mask.shape)+[3])
        else:
            repeat = [ 1 for i in range(mask.ndim + 1) ]
            repeat[-1] = 3
            mask = mask.unsqueeze(-1).repeat(*repeat)
        
    else:
        breakpoint()
    
    return mask
    
def fundus_map_mask(mask, exclusive=False):
    num_classes = 3
    nhot_shape = list(mask.shape)
    if len(nhot_shape) == 2:
        nhot_shape.insert(0, num_classes)
    nhot_shape[-3] = num_classes
    
    if type(mask) == torch.Tensor:
        mask_nhot = torch.zeros(nhot_shape, device=mask.device)
    else:
        mask_nhot = np.zeros(nhot_shape)
    
    # Has the batch dimension
    if mask.ndim == 4:
        # Fake mask. No groundtruth mask available.
        if mask.shape[1] == 1:
            return mask_nhot

        mask_nhot[:, 0] = (mask[:, 0] == 0)     # 0        in channel 0 is background.
        if not exclusive:
            mask_nhot[:, 1] = (mask[:, 0] >= 1)                     # 1 or 255 in channel 0 is optic disc AND optic cup.
        else:
            mask_nhot[:, 1] = (mask[:, 0] >= 1) & (mask[:, 1] == 0) # 1 or 255 in channel 0, excluding 1/255 in channel 1 is optic disc only.
        mask_nhot[:, 2] = (mask[:, 1] >= 1)     # 1 or 255 in channel 1 is optic cup.
    # No batch dimension
    elif mask.ndim == 3:
        # Fake mask. No groundtruth mask available.
        if mask.shape[0] == 1:
            return mask_nhot

        mask_nhot[0] = (mask[0] == 0)           # 0        in channel 0 is background.
        if not exclusive:
            mask_nhot[1] = (mask[0] >= 1)                   # 1 or 255 in channel 0 is optic disc AND optic cup.
        else:
            mask_nhot[1] = (mask[0] >= 1) & (mask[1] == 0)  # 1 or 255 in channel 0, excluding 1/255 in channel 1 is optic disc only.
        mask_nhot[2] = (mask[1] >= 1)           # 1 or 255 in channel 1 is optic cup.
    # Convert REFUGE official annotation format to onehot encoding.
    elif mask.ndim == 2:
        # Fake mask. No groundtruth mask available.
        if mask.shape[0] == 1:
            return mask_nhot

        mask_nhot[0] = (mask == 255)                    # 255 (white) in channel 0 is background.
        if not exclusive:
            mask_nhot[1] = (mask <= 128)                # 128 or 0 is optic disc AND optic cup.
        else:
            mask_nhot[1] = mask[0] == 128               # 128 is optic disc only.
        mask_nhot[2] = (mask == 0)                      # 0 is optic cup.

    return mask_nhot

# fundus_inv_map_mask is not the inverse function of fundus_map_mask.
# It maps model prediction to the REFUGE official annotation format.
# fundus_inv_map_mask works no matter whether mask_nhot is in the exclusive format or not.
def fundus_inv_map_mask(mask_nhot):
    num_classes = 3
    # original mask = 255: background
    # original mask = 128: optic disc (including optic cup)
    # original mask = 0:   optic cup
    
    # Has the batch dimension
    if len(mask_nhot.shape) == 4:
        if type(mask_nhot) == np.ndarray:
            mask = np.zeros_like(mask_nhot[:, 0],    dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_nhot[:, 0], dtype=torch.uint8)
        mask[ mask_nhot[:, 0] == 1 ] = 255
        mask[ mask_nhot[:, 1] == 1 ] = 128
        mask[ mask_nhot[:, 2] == 1 ] = 0
    # Single image, no batch dimension
    elif len(mask_nhot.shape) == 3:
        if type(mask_nhot) == np.ndarray:
            mask = np.zeros_like(mask_nhot[0],    dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_nhot[0], dtype=torch.uint8)
        mask[ mask_nhot[0] == 1 ] = 255
        mask[ mask_nhot[1] == 1 ] = 128
        mask[ mask_nhot[2] == 1 ] = 0
    else:
        breakpoint()
    
    return mask

# Convert a soft segmap to a hard segmap, and at the same time:
# make sure background seg map is consistent with the maps of other classes, 
# in case a pixel is predicted with >=0.5 probs both as background and as other classes.
# mask_soft are n-hot soft predictions.
# mask_soft: (batch, channel, h, w) or (channel, h, w)
def harden_segmap2d(mask_soft, T=0.5):
    if type(mask_soft) == np.ndarray:
        mask_hard = (mask_soft >= T).astype(int)
        dim_key = 'axis'
    else:
        mask_hard = (mask_soft >= T).int()
        dim_key = 'dim'
        
    on_dim1 = {dim_key: 1}
    on_dim0 = {dim_key: 0}
            
    if len(mask_hard.shape) == 4:
        mask_hard[:, 0]  = (mask_hard[:, 1:].sum(**on_dim1) == 0)
    elif len(mask_hard.shape) == 3:
        mask_hard[0]  = (mask_hard[1:].sum(**on_dim0) == 0)
    else:
        breakpoint()
         
    return mask_hard

# polyp mask has one channel repeated 3 times. But only the first channel is referred here.
# Only two classes. The argument 'exclusive' has no effect.
def polyp_map_mask(mask, exclusive=True):
    num_classes = 2
    # The 3-rd channel is unnecessary. Will remove later. 
    # (Not generating 2-channel tensor here to avoid complex logic)

    if type(mask) == torch.Tensor:
        mask_nhot = torch.zeros(mask.shape, device=mask.device)
    else:
        mask_nhot = np.zeros(mask.shape)
    
    # Has the batch dimension
    if mask.ndim == 4:
        mask_nhot[:, 0] = (mask[:, 0] == 0)      # 0   in channel 0 is background.
        mask_nhot[:, 1] = (mask[:, 0] >  0)     # 255 in channel 0 is polyp.
        mask_nhot = mask_nhot[:, :2]
    # No batch dimension
    elif mask.ndim == 3:
        mask_nhot[0] = (mask[0] == 0)        # 0   in channel 0 is background.
        mask_nhot[1] = (mask[0] >  0)       # 255 in channel 0 is polyp.
        mask_nhot = mask_nhot[:2]
    else:
        breakpoint()
    
    return mask_nhot

def polyp_inv_map_mask(mask_nhot):
    num_classes = 2
    # original mask = 255: polyp
    # original mask = 0:   background
    
    # Has the batch dimension
    if len(mask_nhot.shape) == 4:
        if type(mask_nhot) == np.ndarray:
            mask = np.zeros_like(mask_nhot[:, 0],    dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_nhot[:, 0], dtype=torch.uint8)
        mask[ mask_nhot[:, 0] == 1 ] = 0
        mask[ mask_nhot[:, 1] == 1 ] = 255
    # Single image, no batch dimension
    elif len(mask_nhot.shape) == 3:
        if type(mask_nhot) == np.ndarray:
            mask = np.zeros_like(mask_nhot[0],    dtype=np.uint8)
        else:
            mask = torch.zeros_like(mask_nhot[0], dtype=torch.uint8)
        mask[ mask_nhot[0] == 1 ] = 0
        mask[ mask_nhot[1] == 1 ] = 255
    else:
        breakpoint()
    
    return mask

# Assume there's only one connected component in the mask
# shape could be rectangle, ellipse (to be implemented)
def reshape_mask(mask, dim, value=255, shape=None):
    # Default do nothing.
    if shape is None:
        return mask
        
    # (y, x) coordinates of foreground masks
    fg_pos      = np.nonzero(mask[:, :, dim] == value)
    fg_pos_xy   = np.stack(fg_pos[::-1], axis=1)
    if shape == 'rectangle':
        points = cv2.boxPoints(cv2.minAreaRect(fg_pos_xy)).astype(int)
    else:
        breakpoint()
    mask2 = np.zeros(mask.shape)
    cv2.fillPoly(mask2, [points], value)
    mask3 = mask.copy()
    # fillPoly only fills in channel 0 of mask2.
    mask3[:, :, dim] = mask2[:, :, 0]
    return mask3

def load_gamma_labels(gamma_label_path):
    LABELS = open(gamma_label_path)
    reader = csv.reader(LABELS)
    image2label = {}
    
    # Skip csv header.
    next(reader)
    
    # 0002,1,0,0
    for row in reader:
        image_name = row[0]
        onehot_labels = np.array([ int(label) for label in row[1:] ])
        label = onehot_labels.argmax()
        image2label[image_name] = label
    
    return image2label
    
def localize(image, mask, min_output_size):
    if type(min_output_size) == int:
        H = W = min_output_size
    else:
        H, W = min_output_size
            
    tempL = np.nonzero(mask)
    # Find the boundary of non-zero mask
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])

    # px, py, pz ensure the output image is at least of min_output_size
    px = max(H - (maxx - minx), 0) // 2
    py = max(W - (maxy - miny), 0) // 2
    # randint(10, 20) lets randomly-sized zero margins included in the output image
    minx = max(minx - np.random.randint(10, 20) - px, 0)
    maxx = min(maxx + np.random.randint(10, 20) + px, H)
    miny = max(miny - np.random.randint(10, 20) - py, 0)
    maxy = min(maxy + np.random.randint(10, 20) + py, W)

    image  = image[minx:maxx, miny:maxy]
    mask = mask[minx:maxx, miny:maxy]
    return image, mask

def load_mask(mask_path, binarize):
    mask_obj = Image.open(mask_path, 'r')
    # Pixels in mask areas have value 255. 
    # After ToTensor() in segmap_trans_func, they become 1.0.
    mask = np.array(mask_obj)
    if binarize:
        # Only allow two choices of values: 0 and 255.
        mask[mask < 255] = 0
        mask_frac = (mask==255).sum() * 1.0 / mask.size    
        
        # This is a dirty fix for polyp dataset. Need to make the interface more sensible later.
        if mask.ndim == 2:
            mask = np.tile(mask, (3,1,1)).transpose([1, 2, 0])
    # Cannot swap axes of masks here. Otherwise common_aug_func() in SegCrop/SegWhole will turn the mask into garbage.
    return mask
                            
class SegCrop(Dataset):
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    # mask_num_classes should be set when there are no groundtruth mask (e.g. on test data),
    # so that fake mask of a correct shape will be returned.
    # chosen_size: if not None, use images of this size only.
    # uncropped_size is needed when we want to generate segmentation maps of the same size as the original images.
    def __init__(self, base_dir, split, mode, sample_num=-1, 
                 mask_num_classes=2, has_mask=True, ds_weight=1.,
                 common_aug_func=None, image_trans_func=None, image_trans_func2=None,
                 segmap_trans_func=None, binarize=False,
                 train_loc_prob=0, chosen_size=None, uncropped_size=None, min_output_size=None,
                 orig_dir=None, orig_ext=None):
        super(SegCrop, self).__init__()
        self._base_dir = base_dir
        self.orig_dir  = orig_dir
        self.orig_ext  = orig_ext
        self.split = split
        self.mode = mode
        self.mask_num_classes = mask_num_classes
        self.has_mask = has_mask
        # If mode==train then has_mask has to be True.
        # assert (self.mode == 'test' or self.has_mask)
        self.common_aug_func    = common_aug_func
        self.image_trans_func   = image_trans_func
        self.image_trans_func2  = image_trans_func2
        self.segmap_trans_func  = segmap_trans_func
        self.binarize           = binarize
        self.train_loc_prob     = train_loc_prob
        self.chosen_size        = chosen_size
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)

            if uncropped_size is -1:
                self.uncropped_size = -1
            else:
                self.uncropped_size = torch.tensor(uncropped_size)
            
        self.min_output_size    = min_output_size
        self.num_modalities     = 0
        self.ds_weight          = torch.tensor(ds_weight, dtype=float)
        
        alllist_filepath   = self._base_dir + '/all.list'
        # If sample_num > 0, train_test_split_frac_or_shot is the number of shots for few-shot learning.
        if sample_num > 0:
            train_test_split_frac_or_shot = sample_num
            trainlist_filepath = self._base_dir + '/train-{}shot.list'.format(sample_num)
            testlist_filepath  = self._base_dir + '/test-{}shot.list'.format(sample_num)
        # Traditional train/test list files.
        else:
            train_test_split_frac_or_shot = 0.85
            trainlist_filepath = self._base_dir + '/train.list'
            testlist_filepath  = self._base_dir + '/test.list'
            
        if not os.path.isfile(trainlist_filepath):
            self.create_file_list(alllist_filepath, trainlist_filepath, testlist_filepath, 
                                  train_test_split_frac_or_shot)
            
        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath,  'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath,  'r') as f:
            self.all_image_list = f.readlines()
            
        if self.split == 'train':
            image_list = self.train_image_list
        elif self.split == 'test':
            image_list = self.test_image_list
        elif self.split == 'all':
            image_list = self.all_image_list
        else:
            breakpoint()
                        
        image_list = [ item.replace('\n', '') for item in image_list ]
        if self.chosen_size:
            image_list2 = filter(lambda name: '_{}_'.format(self.chosen_size) in name, image_list)
            image_list2 = list(image_list2)
        else:
            image_list2 = image_list
            
        self.image_list = image_list2
        self.sample_num = len(self.image_list)
        
        if self.chosen_size:
            print("'{}' {} samples of size {} chosen (total {}) in '{}'".format(
                        self.split, self.sample_num, self.chosen_size, len(image_list), base_dir))
        else:    
            print("'{}' {} samples in '{}'".format(self.split, self.sample_num, base_dir))

    def __len__(self):
        return self.sample_num
        
    def __getitem__(self, idx):
        # images/n0107_800_591,206.png
        image_name      = self.image_list[idx]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            
            if self.uncropped_size is -1:
                image_name2     = get_filename(image_name)
                image_trunk     = image_name2.split("_")[0]
                orig_image_path = os.path.join(self._base_dir, "..", self.orig_dir, image_trunk + self.orig_ext)
                orig_image_obj  = Image.open(orig_image_path, 'r')
                orig_image      = np.array(orig_image_obj)
                uncropped_size  = torch.tensor(orig_image.shape[:-1])
            else:
                uncropped_size = self.uncropped_size
        
        crop_pos_mat    = re.search(r"(\d+),(\d+)", image_name)
        crop_pos        = crop_pos_mat.group(1), crop_pos_mat.group(2)
        crop_pos        = torch.tensor([ int(x) for x in crop_pos ])
        image_path      = os.path.join(self._base_dir, image_name)
        assert ('images' in image_path)
        image_obj       = Image.open(image_path, 'r')
        image           = np.array(image_obj)

        if self.has_mask:
            # masks/n0107_800_591,206.png
            mask_name = image_name.replace("images", "masks")
            assert ('masks' in mask_name)
            mask_path = os.path.join(self._base_dir, mask_name)
            mask = load_mask(mask_path, self.binarize)
        else:
            # Fake mask with all zeros. 'class' is always at the last dimension.
            mask = np.zeros( (1,) + image.shape[:2], dtype=np.uint8 )
            mask_path = ""
            
        unscaled_size = torch.tensor(image.shape[:2])
            
        if self.mode == 'train' and self.train_loc_prob > 0 \
          and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)
        
        if self.common_aug_func:
            # Always mask == segmap.get_arr() == segmap.arr.astype(np.uint8).
            segmap = SegmentationMapsOnImage(mask, shape=image.shape)
            image_aug, segmap_aug = self.common_aug_func(image=image, segmentation_maps=segmap)
            image = image_aug
            mask = segmap_aug.get_arr()
        
        mask = mask.astype(np.uint8)
            
        if self.image_trans_func:
            image_obj  = Image.fromarray(image)
            image1     = self.image_trans_func(image_obj)
        else:
            image1     = image
            
        if self.image_trans_func2:
            image_obj = Image.fromarray(image)
            image2     = self.image_trans_func2(image_obj)
        else:
            image2     = image
        
        if self.segmap_trans_func:
            mask = self.segmap_trans_func(mask)
        else:
            mask = mask.transpose([2, 0, 1])

        sample = { 'image': image1, 'image2': image2, 
                   'mask': mask, 'index': idx,
                   'image_path': image_path,  'mask_path': mask_path,
                   'crop_pos': crop_pos,
                   'unscaled_size': unscaled_size,
                   'uncropped_size': uncropped_size,
                   'weight': self.ds_weight }
        return sample

    def create_file_list(self, alllist_filepath, trainlist_filepath, testlist_filepath, 
                         train_test_split_frac_or_shot=0.85):
        img_list = [ d for d in listdir(join(self._base_dir, 'images')) ]

        self.idx2filenames = {}
        
        for img_filename in img_list:
            img_idx, img_size, crop_pos = img_filename.split("_")
                
            if img_idx not in self.idx2filenames:
                self.idx2filenames[img_idx] = []
            self.idx2filenames[img_idx].append(join('images', img_filename))
        
        # Randomize the file list, then split. Not to use the official testing set 
        # since we don't have ground truth masks for this.
        indices = list(self.idx2filenames.keys())  # List of file unique indices
        num_files = len(indices)

        for img_idx in indices:
            self.idx2filenames[img_idx] = sorted(self.idx2filenames[img_idx])

        with open(alllist_filepath, "w") as allFile:
            for img_idx in indices:
                allFile.write("%s\n" %"\n".join(self.idx2filenames[img_idx]))
        allFile.close()
                
        indices = np.random.permutation(indices)  # Randomize list
        if type(train_test_split_frac_or_shot) == int:
            train_len = train_test_split_frac_or_shot
        else:
            # Number of training files
            train_len = int(np.floor(num_files * train_test_split_frac_or_shot))
            
        train_indices = indices[0:train_len]  # List of training indices
        test_indices  = indices[train_len:]  # List of testing indices

        with open(trainlist_filepath, "w") as trainFile:
            for img_idx in sorted(train_indices):
                trainFile.write("%s\n" %"\n".join(self.idx2filenames[img_idx]))
        trainFile.close()
        
        with open(testlist_filepath,  "w") as testFile:
            for img_idx in sorted(test_indices):
                testFile.write("%s\n"  %"\n".join(self.idx2filenames[img_idx]))
        testFile.close()
        
        print("%d files are split to %d training, %d test" %(num_files, train_len, len(test_indices)))


class SegWhole(Dataset):
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    # mask_num_classes should be set when there are no groundtruth mask (e.g. on test data),
    # so that fake mask of a correct shape will be returned.
    # chosen_size, uncropped_size are useless and ignored. Just to keep conformed with SegCrop.
    def __init__(self, base_dir, split, mode, sample_num=-1, 
                 mask_num_classes=2, has_mask=True, ds_weight=1.,
                 common_aug_func=None, image_trans_func=None,
                 segmap_trans_func=None, binarize=False,
                 train_loc_prob=0, chosen_size=None, 
                 uncropped_size=None, min_output_size=None,
                 orig_dir=None, orig_ext=None):
        super(SegWhole, self).__init__()
        self._base_dir = base_dir
        self.split = split
        self.mode = mode
        self.mask_num_classes = mask_num_classes
        self.has_mask = has_mask
        # If mode==train then has_mask has to be True.
        # assert (self.mode == 'test' or self.has_mask)
        self.common_aug_func    = common_aug_func
        self.image_trans_func   = image_trans_func
        self.segmap_trans_func  = segmap_trans_func
        self.binarize           = binarize
        self.train_loc_prob     = train_loc_prob
        self.min_output_size    = min_output_size
        self.num_modalities     = 0
        self.ds_weight          = torch.tensor(ds_weight, dtype=float)
        
        alllist_filepath   = self._base_dir + '/all.list'
        # If sample_num > 0, train_test_split_frac_or_shot is the number of shots for few-shot learning.
        if sample_num > 0:
            train_test_split_frac_or_shot = sample_num
            trainlist_filepath = self._base_dir + '/train-{}shot.list'.format(sample_num)
            testlist_filepath  = self._base_dir + '/test-{}shot.list'.format(sample_num)
        # Traditional train/test list files.
        else:
            train_test_split_frac_or_shot = 0.85
            trainlist_filepath = self._base_dir + '/train.list'
            testlist_filepath  = self._base_dir + '/test.list'
        
        if not os.path.isfile(trainlist_filepath):
            self.create_file_list(alllist_filepath, trainlist_filepath, testlist_filepath, 
                                  train_test_split_frac_or_shot)
            
        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath,  'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath,  'r') as f:
            self.all_image_list = f.readlines()
            
        if self.split == 'train':
            image_list = self.train_image_list
        elif self.split == 'test':
            image_list = self.test_image_list
        elif self.split == 'all':
            image_list = self.all_image_list
        else:
            breakpoint()
            
        self.image_list = [ item.replace('\n', '') for item in image_list]
        self.sample_num = len(self.image_list)
        
        print("'{}' {} samples in '{}'".format(self.split, self.sample_num, base_dir))

    def __len__(self):
        return self.sample_num
        
    def __getitem__(self, idx):
        # images/101.png
        image_name      = self.image_list[idx]
        image_path      = os.path.join(self._base_dir, image_name)
        image_obj       = Image.open(image_path, 'r')
        image           = np.array(image_obj)
        if image.ndim == 2:
            image = np.expand_dims(image, 2).repeat(3, 2)
        if image.shape[2] == 1:
            image = image.repeat(3, 2)
            
        if self.has_mask:
            # masks/101.png
            mask_name = image_name.replace("images", "masks")
            mask_path = os.path.join(self._base_dir, mask_name)
            mask = load_mask(mask_path, self.binarize)
        else:
            # Fake mask with all zeros. 'class' is always at the last dimension.
            mask = np.zeros( (1,) + image.shape[:2], dtype=np.uint8 )
            mask_path = ""
            
        unscaled_size = torch.tensor(image.shape[:2])
            
        if self.mode == 'train' and self.train_loc_prob > 0 \
          and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)
        
        if self.common_aug_func:
            # Always mask == segmap.get_arr() == segmap.arr.astype(np.uint8).
            segmap = SegmentationMapsOnImage(mask, shape=image.shape)
            image_aug, segmap_aug = self.common_aug_func(image=image, segmentation_maps=segmap)
            image = image_aug
            mask = segmap_aug.get_arr()
        
        mask = mask.astype(np.uint8)
            
        if self.image_trans_func:
            image_obj = Image.fromarray(image)
            image     = self.image_trans_func(image_obj)
            
        if self.segmap_trans_func:
            mask = self.segmap_trans_func(mask)
        else:
            mask = mask.transpose([2, 0, 1])
            
        sample = { 'image': image, 'mask': mask, 
                   'image_path': image_path, 'mask_path': mask_path,
                   'crop_pos': torch.tensor((-1, -1)),
                   'unscaled_size': unscaled_size,
                   'uncropped_size': torch.tensor((-1, -1)),
                   'weight': self.ds_weight  }
        return sample

    def create_file_list(self, alllist_filepath, trainlist_filepath, testlist_filepath, 
                         train_test_split_frac_or_shot=0.85):
        img_list = [ d for d in listdir(join(self._base_dir, 'images')) ]

        self.idx2filenames = {}
        
        for img_filename in img_list:
            img_idx = img_filename
            if img_idx not in self.idx2filenames:
                self.idx2filenames[img_idx] = []
            self.idx2filenames[img_idx].append(join('images', img_filename))
        
        # Randomize the file list, then split. Not to use the official testing set 
        # since we don't have ground truth masks for this.
        indices = list(self.idx2filenames.keys())  # List of file unique indices
        num_files = len(indices)

        for img_idx in indices:
            self.idx2filenames[img_idx] = sorted(self.idx2filenames[img_idx])

        with open(alllist_filepath, "w") as allFile:
            for img_idx in indices:
                allFile.write("%s\n" %"\n".join(self.idx2filenames[img_idx]))
        allFile.close()
                
        indices = np.random.permutation(indices)  # Randomize list
        if type(train_test_split_frac_or_shot) == int:
            train_len = train_test_split_frac_or_shot
        else:
            # Number of training files
            train_len = int(np.floor(num_files * train_test_split_frac_or_shot))
            
        train_indices = indices[0:train_len]  # List of training indices
        test_indices  = indices[train_len:]   # List of testing indices

        with open(trainlist_filepath, "w") as trainFile:
            for img_idx in sorted(train_indices):
                trainFile.write("%s\n" %"\n".join(self.idx2filenames[img_idx]))
        trainFile.close()
        
        with open(testlist_filepath,  "w") as testFile:
            for img_idx in sorted(test_indices):
                testFile.write("%s\n"  %"\n".join(self.idx2filenames[img_idx]))
        testFile.close()
        
        print("%d files are split to %d training, %d test" %(num_files, train_len, len(test_indices)))
