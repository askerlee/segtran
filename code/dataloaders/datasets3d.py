import os
from os import listdir
from os.path import isdir, join
import re
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import json
import pdb

def brats_map_label(mask, binarize):
    if binarize:
        num_classes = 2
    else:
        num_classes = 4
        
    if type(mask) == torch.Tensor:
        mask_nhot = torch.zeros((num_classes,) + mask.shape, device='cuda')
    else:
        mask_nhot = np.zeros((num_classes,) + mask.shape)

    if binarize:
        mask_nhot[0, mask==0] = 1
        mask_nhot[1, mask>0]  = 1  
    else:
        # 1 for NCR & NET, 2 for ED, 3 for ET, and 0 for everything else.
        mask_nhot[0, mask==0] = 1
        mask_nhot[1, mask==3] = 1                               # P(ET) = P(3)
        mask_nhot[2, (mask==3) | (mask==1) | (mask==2)] = 1 # P(WT) = P(1)+P(2)+P(3)
        mask_nhot[3, (mask==3) | (mask==1)] = 1               # P(TC) = P(1)+P(3)
        # Has the batch dimension. Swap the batch to the zero-th dim.

    if len(mask_nhot.shape) == 5:
        mask_nhot = mask_nhot.permute(1, 0, 2, 3, 4)
    return mask_nhot

# Called only when predicting each class separately
def make_brats_pred_consistent(preds_soft, is_conservative):
    # is_conservative: predict 0 as much as it can. 
    # Predict 1 only when predicting 1 on all superclasses.
    # WT overrules TC; TC overrules ET. 1: ET, 2: WT, 3: TC.
    preds_soft2 = preds_soft.clone()
    if is_conservative:
        # If not WT or not TC, then not ET => P(ET) = min(P(ET), P(TC), P(WT))
        # # If not WT, then not TC         => P(TC) = min(P(TC), P(WT))
        preds_soft2[1]  = torch.min(preds_soft[1:],  dim=0)[0]
        preds_soft2[3]  = torch.min(preds_soft[2:],  dim=0)[0]    
    # Predict 1, as long as predicting 1 on one of its subclasses.
    # ET overrules TC; TC overrules WT.
    else:
        # If TC then WT => P(WT) >= P(TC)
        preds_soft2[2]  = torch.max(preds_soft[1:],    dim=0)[0]
        # If ET then TC => P(TC) >= P(ET)
        preds_soft2[3]  = torch.max(preds_soft[[1,3]], dim=0)[0]

    return preds_soft2
    
def brats_inv_map_label(orig_probs):
    # orig_probs[0] is not used. Prob of 0 (background) is inferred from probs of WT.
    if type(orig_probs) == torch.Tensor:
        inv_probs = torch.zeros_like(orig_probs)
    else:
        inv_probs = np.zeros_like(orig_probs)
        
    inv_probs[0] = 1 - orig_probs[2]                 # P(0) = 1 - P(WT)
    inv_probs[3] = orig_probs[1]                     # P(3) = P(ET)
    UP = 1.5            # Slightly increase the prob of predicting 1 and 2.
    inv_probs[1] = orig_probs[3] - orig_probs[1]     # P(1) = P(TC) - P(ET)
    inv_probs[1] *= UP
    inv_probs[2] = orig_probs[2] - orig_probs[3]     # P(2) = P(WT) - P(TC)
    inv_probs[2] *= UP
    
    if (inv_probs < 0).sum() > 0:
        pdb.set_trace()
        
    '''
    mask_nhot[0, mask==0] = 1
    mask_nhot[1, mask==3] = 1                               # P(ET) = P(3)
    mask_nhot[2, (mask==3) | (mask==1) | (mask==2)] = 1 # P(WT) = P(1)+P(2)+P(3)
    mask_nhot[3, (mask==3) | (mask==1)] = 1               # P(TC) = P(1)+P(3)
    '''
    
    return inv_probs

# Convert a soft segmap to a hard segmap, and at the same time:
# make sure background seg map is consistent with the maps of other classes, 
# in case a pixel is predicted with >=0.5 probs both as background and as other classes.
# mask_soft are n-hot soft predictions.
# mask_soft: (batch, channel, h, w) or (channel, h, w)
def harden_segmap3d(mask_soft, T=0.5):
    if type(mask_soft) == np.ndarray:
        mask_hard = (mask_soft >= T).astype(int)
        dim_key = 'axis'
    else:
        mask_hard = (mask_soft >= T).int()
        dim_key = 'dim'
        
    on_dim1 = {dim_key: 1}
    on_dim0 = {dim_key: 0}
            
    if len(mask_hard.shape) == 5:
        mask_hard[:, 0]  = (mask_hard[:, 1:].sum(**on_dim1) == 0)
    elif len(mask_hard.shape) == 4:
        mask_hard[0]  = (mask_hard[1:].sum(**on_dim0) == 0)
    else:
        pdb.set_trace()
         
    return mask_hard
        
def localize(image, mask, min_output_size):
    if type(min_output_size) == int:
        H = W = D = min_output_size
    else:
        H, W, D = min_output_size
            
    tempL = np.nonzero(mask)
    # Find the boundary of non-zero mask
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    # px, py, pz ensure the output image is at least of min_output_size
    px = max(min_output_size[0] - (maxx - minx), 0) // 2
    py = max(min_output_size[1] - (maxy - miny), 0) // 2
    pz = max(min_output_size[2] - (maxz - minz), 0) // 2
    # randint(10, 20) lets randomly-sized zero margins included in the output image
    minx = max(minx - np.random.randint(10, 20) - px, 0)
    maxx = min(maxx + np.random.randint(10, 20) + px, H)
    miny = max(miny - np.random.randint(10, 20) - py, 0)
    maxy = min(maxy + np.random.randint(10, 20) + py, W)
    minz = max(minz - np.random.randint(5, 10) - pz, 0)
    maxz = min(maxz + np.random.randint(5, 10) + pz, D)

    if len(image.shape) == 4:                
        image = image[:, minx:maxx, miny:maxy, minz:maxz]
    else:
        image = image[minx:maxx, miny:maxy, minz:maxz]
            
    mask = mask[minx:maxx, miny:maxy, minz:maxz]
    return image, mask
    
class AtriaSet(Dataset):
    """ LA Dataset """
    # LA mask are always binarize. 
    # The arguments 'chosen_modality' and 'binarize' are just placeholders.
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    def __init__(self, base_dir, split, mode, sample_num=None, 
                 transform=None, chosen_modality=-1, binarize=True,
                 train_loc_prob=0, min_output_size=None):
        super(AtriaSet, self).__init__()
        self._base_dir = base_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.binarize    = binarize
        self.num_modalities = 0
        self.train_loc_prob = train_loc_prob
        self.min_output_size = min_output_size

        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath,  'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath,  'r') as f:
            self.all_image_list = f.readlines()
            
        if self.split == 'train':
            self.image_list = self.train_image_list
        elif self.split == 'test':
            self.image_list = self.test_image_list
        elif self.split == 'all':
            self.image_list = self.all_image_list
                
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if sample_num is not None:
            self.image_list = self.image_list[:sample_num]
        print("{} {} samples".format(self.split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        mask  = h5f['label'][:]

        if self.mode == 'train' and self.train_loc_prob > 0 \
          and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)
            
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

class MSDSet(Dataset):
    """ Medical Segmentation Decathlon task dataset """
    # binarize: whether to binarize mask
    # modality: if the image has multiple modalities, 
    # choose which modality to output (-1 to output all)
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    def __init__(self, base_dir, split, mode, sample_num=None, 
                 transform=None, chosen_modality=-1, binarize=False,
                 train_loc_prob=0, min_output_size=None):
        super(MSDSet, self).__init__()
        self._base_dir = base_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.chosen_modality = chosen_modality
        self.binarize    = binarize
        self.train_loc_prob = train_loc_prob
        self.min_output_size = min_output_size
        
        trainlist_filepath = self._base_dir + '/train.list'
        testlist_filepath  = self._base_dir + '/test.list'
        if not os.path.isfile(trainlist_filepath):
            self.create_file_list(0.85)

        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath,  'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath,  'r') as f:
            self.all_image_list = f.readlines()
            
        if self.split == 'train':
            self.image_list = self.train_image_list
        elif self.split == 'test':
            self.image_list = self.test_image_list
        elif self.split == 'all':
            self.image_list = self.all_image_list

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if sample_num is not None:
            self.image_list = self.image_list[:sample_num]
                
        self.num_modalities = 0
        # Fetch image 0 to get num_modalities.
        sample0 = self.__getitem__(0, do_transform=False)
        if len(sample0['image'].shape) == 4:
            self.num_modalities = sample0['image'].shape[3]
            
        print("{} {} samples, num_modalities: {}, output: {}".format(self.split, 
                len(self.image_list), self.num_modalities, self.chosen_modality))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, do_transform=True):
        image_name = self.image_list[idx]
        image_path = os.path.join(self._base_dir, image_name)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        mask  = h5f['label'][:]
        if self.num_modalities > 0 and self.chosen_modality != -1:
            image = image[self.chosen_modality, :, :, :]
        if self.binarize:
            mask = (mask >= 1).astype(np.uint8)

        if self.mode == 'train' and self.train_loc_prob > 0 \
          and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)
            
        sample = {'image': image, 'mask': mask}
        if do_transform and self.transform:
            sample = self.transform(sample)

        return sample

    def create_file_list(self, train_test_split):
        # Get list of the files from the MSD json file, split them into training and test sets.
        json_filename = os.path.join(self._base_dir, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be part of the "
                  "Decathlon directory".format(json_filename))

        # Randomize the file list, then split. Not to use the official testing set 
        # since we don't have ground truth masks for this.
        num_files = experiment_data["numTraining"]
        idxList = np.arange(num_files)  # List of file indices
        self.imgFiles = {}
        for idx in idxList:
            self.imgFiles[idx] = experiment_data["training"][idx]["image"]
            self.imgFiles[idx] = self.imgFiles[idx].replace(".nii.gz", ".h5")
            if self.imgFiles[idx][:2] == './':
                self.imgFiles[idx] = self.imgFiles[idx][2:]
                
        idxList = np.random.permutation(idxList)  # Randomize list
        train_len = int(np.floor(num_files * train_test_split)) # Number of training files
        train_indices = idxList[0:train_len]  # List of training indices
        test_indices  = idxList[train_len:]  # List of testing indices

        with open( join(self._base_dir, 'train.list'), "w" ) as trainFile:
            for img_idx in sorted(train_indices):
                trainFile.write("%s\n" %self.imgFiles[img_idx])
        trainFile.close()
        
        with open( join(self._base_dir, 'test.list'),  "w" ) as testFile:
            for img_idx in sorted(test_indices):
                testFile.write("%s\n"  %self.imgFiles[img_idx])
        testFile.close()
        
        print("%d files are split to %d training, %d test" %(num_files, train_len, len(test_indices)))

class BratsSet(Dataset):
    """ Annual Brats challenges dataset """
    # binarize: whether to binarize mask (do whole-tumor segmentation)
    # modality: if the image has multiple modalities, 
    # choose which modality to output (-1 to output all).
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    def __init__(self, base_dir, split, mode, sample_num=None, 
                 mask_num_classes=2, has_mask=True, ds_weight=1.,
                 xyz_permute=None, transform=None, 
                 chosen_modality=-1, binarize=False,
                 train_loc_prob=0, min_output_size=None):
        super(BratsSet, self).__init__()
        self._base_dir = base_dir
        self.split = split
        self.mode = mode
        self.mask_num_classes = mask_num_classes
        self.has_mask = has_mask
        # If mode==train then has_mask has to be True.
        # assert (self.mode == 'test' or self.has_mask)
        self.xyz_permute = xyz_permute
        self.ds_weight = ds_weight
        
        self.transform = transform
        self.chosen_modality = chosen_modality
        self.binarize    = binarize
        self.train_loc_prob = train_loc_prob
        self.min_output_size = min_output_size
        
        trainlist_filepath = self._base_dir + '/train.list'
        testlist_filepath  = self._base_dir + '/test.list'
        alllist_filepath   = self._base_dir + '/all.list'
        
        if not os.path.isfile(alllist_filepath):
            self.create_file_list(0.85)
            
        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath,  'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath,  'r') as f:
            self.all_image_list = f.readlines()

        if self.split == 'train':
            self.image_list = self.train_image_list
        elif self.split == 'test':
            self.image_list = self.test_image_list
        elif self.split == 'all':
            self.image_list = self.all_image_list

        self.image_list = [ item.replace('\n', '') for item in self.image_list ]
        if sample_num is not None:
            self.image_list = self.image_list[:sample_num]
                
        self.num_modalities = 0
        # Fetch image 0 to get num_modalities.
        sample0 = self.__getitem__(0, do_transform=False)
        if len(sample0['image'].shape) == 4:
            self.num_modalities = sample0['image'].shape[0]
        
        print("'{}' {} samples, num_modalities: {}, chosen: {}".format(self.split, 
                len(self.image_list), self.num_modalities, self.chosen_modality))

    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, idx, do_transform=True):
        image_name = self.image_list[idx]
        image_path = os.path.join(self._base_dir, image_name)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        mask  = h5f['label'][:]
        if self.num_modalities > 0 and self.chosen_modality != -1:
            image = image[self.chosen_modality, :, :, :]
        if self.binarize:
            mask = (mask >= 1).astype(np.uint8)
        else:
            # Map 4 to 3, and keep 0,1,2 unchanged.
            mask -= (mask == 4)

        if self.mode == 'train' and self.train_loc_prob > 0 \
          and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)

        # xyz_permute by default is None.
        if do_transform and self.xyz_permute is not None:
            image = image.transpose(self.xyz_permute)
            mask  = mask.transpose(self.xyz_permute)
            
        sample = { 'image': image, 'mask': mask }
        if do_transform and self.transform:
            sample = self.transform(sample)
            
        sample['image_path'] = image_name
        sample['weight'] = self.ds_weight
        return sample

    def create_file_list(self, train_test_split):
        img_dirs = [ d for d in listdir(self._base_dir) if isdir(join(self._base_dir, d)) ]

        # Randomize the file list, then split. Not to use the official testing set 
        # since we don't have ground truth masks for this.
        num_files = len(img_dirs)
        idxList = np.arange(num_files)  # List of file indices
        self.imgFiles = {}
        for idx in idxList:
            self.imgFiles[idx] = join(img_dirs[idx], img_dirs[idx] + ".h5")

        with open( join(self._base_dir, 'all.list'), "w" ) as allFile:
            for img_idx in idxList:
                allFile.write("%s\n" %self.imgFiles[img_idx])
        allFile.close()
                
        idxList = np.random.permutation(idxList)  # Randomize list
        train_len = int(np.floor(num_files * train_test_split)) # Number of training files
        train_indices = idxList[0:train_len]  # List of training indices
        test_indices  = idxList[train_len:]  # List of testing indices

        with open( join(self._base_dir, 'train.list'), "w" ) as trainFile:
            for img_idx in sorted(train_indices):
                trainFile.write("%s\n" %self.imgFiles[img_idx])
        trainFile.close()
        
        with open( join(self._base_dir, 'test.list'),  "w" ) as testFile:
            for img_idx in sorted(test_indices):
                testFile.write("%s\n"  %self.imgFiles[img_idx])
        testFile.close()
        
        print("%d files are split to %d training, %d test" %(num_files, train_len, len(test_indices)))
            
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # pad the sample if necessary
        if mask.shape[0] <= self.output_size[0] or mask.shape[1] <= self.output_size[1] or mask.shape[2] <= \
                self.output_size[2]:
            ph = max((self.output_size[0] - mask.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - mask.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - mask.shape[2]) // 2 + 3, 0)
            mask_padding = [(ph, ph), (pw, pw), (pd, pd)]
            image_padding = mask_padding
            if len(image.shape) == 4:
                image_padding = [(0, 0)] + image_padding

            image = np.pad(image, image_padding, mode='constant', constant_values=0)
            mask  = np.pad(mask,  mask_padding,  mode='constant', constant_values=0)

        (h, w, d) = image.shape[:3]

        h1 = int(round((h - self.output_size[0]) / 2.))
        w1 = int(round((w - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        mask  = mask[h1:h1  + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if len(image.shape) == 4:
            image = image[:, h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
        else:
            image = image[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
                
        return {'image': image, 'mask': mask}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        orig_mask_shape = list(mask.shape)
        
        # pad the sample if necessary
        if mask.shape[0] <= self.output_size[0] or mask.shape[1] <= self.output_size[1] or mask.shape[2] <= \
                self.output_size[2]:
            ph = max((self.output_size[0] - mask.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - mask.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - mask.shape[2]) // 2 + 3, 0)
            mask_padding = [(ph, ph), (pw, pw), (pd, pd)]
            image_padding = mask_padding
            # the modality dimension.
            if len(image.shape) == 4:
                image_padding = [(0, 0)] + image_padding
            
            try:
                image = np.pad(image, image_padding, mode='constant', constant_values=0)
                mask  = np.pad(mask,  mask_padding,  mode='constant', constant_values=0)
            except:
                pdb.set_trace()
        else:
            image_padding = 0
            
        # print(orig_mask_shape, image_padding, mask.shape, self.output_size)
        (h, w, d) = image.shape[-3:]
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        h1 = np.random.randint(0, h - self.output_size[0])
        w1 = np.random.randint(0, w - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        # print(h1, h1 + self.output_size[0], w1, w1 + self.output_size[1], d1, d1 + self.output_size[2])
            
        mask  = mask[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
        # image: (118, 118, 102). mask: (118, 118, 102) => (112, 112, 96)
        if image.ndim == 4:
            # For a 4D image, the first dimension is modality.
            image = image[:, h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
        else:
            image = image[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'mask': mask}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        assert (image.ndim == mask.ndim) or (image.ndim == mask.ndim + 1)
        # Rotate by k*90. k is not the axis.
        k = np.random.randint(0, 4)

        # Image has multiple modalities. First dimension is the modality.
        # Only rotate along the x,y (0,1) plane.
        if image.ndim == mask.ndim + 1:
            image = np.rot90(image, k, axes=(1,2))
        else:
            image = np.rot90(image, k, axes=(0,1))
            
        mask = np.rot90(mask, k, axes=(0,1))
        
        # Flip along a random dimension. It doesn't change the shape of a tensor.
        axis = np.random.randint(0, 3)
        if image.ndim == mask.ndim + 1:
            image = np.flip(image, axis=axis+1).copy()
        else:
            image = np.flip(image, axis=axis).copy()
            
        mask = np.flip(mask, axis=axis).copy()

        return { 'image': image, 'mask': mask }


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1, nonzero_only=True):
        self.mu = mu
        self.sigma = sigma
        self.nonzero_only = nonzero_only
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        noise = np.clip(self.sigma * np.random.randn(*image.shape), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        if self.nonzero_only:
            # Only add noise to non-zero voxels
            image = image + noise * (image != 0)
        else:
            image = image + noise
            
        return {'image': image, 'mask': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        mask  = sample['mask']
        if image.ndim == 3:
            image = image.reshape((1,) + image.shape)
        return { 'image': torch.from_numpy(image), 
                 'mask':  torch.from_numpy(mask) }

# mask should be one-hot. shape: Batch, Class, H, W, D
def RandomResizedCrop(volume, mask, out_size, crop_percents, isotropic=True):
    H, W, D = volume.shape[-3:]
    min_crop, max_crop = crop_percents  # -0.1, 0.1
    min_scale = 1 + min_crop            # 0.9
    max_scale = 1 + max_crop            # 1.1
    
    if not isotropic:
        scale_H = torch.rand(1) * (max_scale - min_scale) + min_scale
        scale_W = torch.rand(1) * (max_scale - min_scale) + min_scale
        scale_D = torch.rand(1) * (max_scale - min_scale) + min_scale
    else:
        # scale_H is uniform between [min_scale, max_scale]
        scale_H = torch.rand(1) * (max_scale - min_scale) + min_scale
        scale_W = scale_D = scale_H

    H2 = int(H * scale_H)
    W2 = int(W * scale_W)
    D2 = int(D * scale_D)
    
    # volume: [4, 1, 112, 112, 96]. mask: [4, 4, 112, 112, 96]
    volume2 = F.interpolate(volume, size=(H2, W2, D2), mode='trilinear', align_corners=False)
    mask2   = F.interpolate(mask,   size=(H2, W2, D2), mode='trilinear', align_corners=False)
    
    # out_size: (112, 112, 96)
    H_out, W_out, D_out = out_size
    if H2 < H_out or W2 < W_out or D2 < D_out:
        pad_h  = max(H_out - H2, 0)
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        pad_w  = max(W_out - W2, 0)
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_d  = max(D_out - D2, 0)
        pad_d1 = pad_d // 2
        pad_d2 = pad_d - pad_d1
        
        pads = (pad_d1, pad_d2, pad_w1, pad_w2, pad_h1, pad_h2)
        
        volume2 = F.pad(volume2, pads, "constant", 0)
        mask2   = F.pad(mask2,   pads, "constant", 0)
    
    H2, W2, D2 = volume2.shape[2:] 
    h_start = torch.randint(H2 - H_out + 1, (1,))
    h_end   = h_start + H_out
    w_start = torch.randint(W2 - W_out + 1, (1,))
    w_end   = w_start + W_out
    d_start = torch.randint(D2 - D_out + 1, (1,))
    d_end   = d_start + D_out
        
    volume3 = volume2[:, :, h_start:h_end, w_start:w_end, d_start:d_end].clone()
    mask3   = mask2[:,   :, h_start:h_end, w_start:w_end, d_start:d_end].clone()
    # If converting the continuous mask values (produced by interpolation) to one-hot, performance drops. So keep the cont
    # mask3   = (mask3 >= 0.5).type(mask.dtype)
    
    return volume3, mask3
    