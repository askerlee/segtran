import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib
import sys
import os
from os import listdir
from os.path import isdir, join
import pdb

output_size =[128, 144, 80]

def brats_map_label(labels, num_classes):
    if type(labels) == torch.Tensor:
        labels_nhot = torch.zeros((num_classes,) + labels.shape, device='cuda')
    else:
        labels_nhot = np.zeros((num_classes,) + labels.shape, dtype=int)
        
    labels_nhot[0, labels==0] = 1
    labels_nhot[1, labels==3] = 1                               # P(ET) = P(3)
    labels_nhot[2, (labels==3) | (labels==1) | (labels==2)] = 1 # P(WT) = P(1)+P(2)+P(3)
    labels_nhot[3, (labels==3) | (labels==1)] = 1               # P(TC) = P(1)+P(3)
    return labels_nhot

def label_stats(root):
    img_dirs = [ d for d in listdir(root) if isdir(join(root, d)) ]
    cls_total_counts = np.zeros(4, dtype=int)
        
    for img_dir in tqdm(img_dirs):
        img_path = join(root, img_dir, img_dir + ".h5")
        image_h5 = h5py.File(img_path)
        labels = np.array(image_h5['label']).astype(int)
        labels -= (labels == 4)
        labels_nhot = brats_map_label(labels, 4)
        cls_counts = labels_nhot.reshape(4, -1).sum(axis=1)
        print(cls_counts)
        cls_total_counts += cls_counts

        tempL = np.nonzero(labels_nhot[2] == 1)
        # Find the boundary of non-zero labels
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        H, W, D = maxx - minx, maxy - miny, maxz - minz
        
        print("%s => %s, %.2f%%" %(labels.shape, (H, W, D), (100*H*W*D/labels.size)))
                
    print(cls_total_counts)
    print(cls_total_counts / cls_total_counts.sum())
    
def covert_h5(root):
    do_localization = False

    if 'validation' in root.lower():
        is_training = False
    else:
        is_training = True
        
    img_dirs = [ d for d in listdir(root) if isdir(join(root, d)) ]
    modalities = [ 'flair', 't1ce', 't1', 't2' ]
    if is_training:
        modalities.append('seg')
        
    for img_dir in tqdm(img_dirs):
        image_mods = []
        for mod in modalities:
            img_path = join(root, img_dir, img_dir + "_%s.nii.gz" %mod)
            image_obj = nib.load(img_path)
            image = image_obj.get_fdata()
            image = image.astype(np.float32)
            image_mods.append(image)
        
        if is_training:
            # image_mods contains five modalities, including 'seg'.
            # Avoid putting 'seg' into 'image'.
            image_mm = np.stack(image_mods[:-1], axis=0)
            MOD, H, W, D = image_shape = image_mm.shape
            # 'seg' => labels
            labels = image_mods[-1].astype(np.uint8)
        else:
            # image_mods contains four modalities
            image_mm = np.stack(image_mods, axis=0)
            MOD, H, W, D = image_shape = image_mm.shape
            # Save fake labels
            labels = np.zeros_like(image_mods[0]).astype(np.uint8)
            
        if is_training and do_localization:
            tempL = np.nonzero(labels)
            # Find the boundary of non-zero labels
            minx, maxx = np.min(tempL[1]), np.max(tempL[1])
            miny, maxy = np.min(tempL[2]), np.max(tempL[2])
            minz, maxz = np.min(tempL[3]), np.max(tempL[3])

            # px, py, pz ensure the output image is at least of output_size
            px = max(output_size[0] - (maxx - minx), 0) // 2
            py = max(output_size[1] - (maxy - miny), 0) // 2
            pz = max(output_size[2] - (maxz - minz), 0) // 2
            # randint(10, 20) lets randomly-sized zero margins included in the output image
            minx = max(minx - np.random.randint(10, 20) - px, 0)
            maxx = min(maxx + np.random.randint(10, 20) + px, H)
            miny = max(miny - np.random.randint(10, 20) - py, 0)
            maxy = min(maxy + np.random.randint(10, 20) + py, W)
            minz = max(minz - np.random.randint(5, 10) - pz, 0)
            maxz = min(maxz + np.random.randint(5, 10) + pz, D)
        # Do not crop black margins of validation images.
        else:
            tempL = np.nonzero(image_mm)
            # Find the boundary of non-zero voxels
            minx, maxx = np.min(tempL[1]), np.max(tempL[1])
            miny, maxy = np.min(tempL[2]), np.max(tempL[2])
            minz, maxz = np.min(tempL[3]), np.max(tempL[3])
            
        # On validation data, use image_crop to compute mean and std, and normalize image_mm.
        image_crop = image_mm[:, minx:maxx, miny:maxy, minz:maxz]
        if is_training:
            image_mm = image_crop
            labels = labels[minx:maxx, miny:maxy, minz:maxz]

        nonzero_mask = (image_mm > 0)
        for m in range(MOD):
            image_mod      = image_mm[m, :, :, :]
            image_mod_crop = image_crop[m, :, :, :]
            nonzero_voxels = image_mod_crop[image_mod_crop > 0]
            mean = nonzero_voxels.mean()
            std  = nonzero_voxels.std()
            image_mm[m, :, :, :] = (image_mod - mean) / std
        
        # Set voxels back to 0 if they are 0 before normalization.
        image_mm *= nonzero_mask
        print("\n%s: %s => %s, %s" %(img_dir, image_shape, image_mm.shape, labels.shape))

        h5_path = join(root, img_dir, img_dir + ".h5")
        f = h5py.File(h5_path, 'w')
        f.create_dataset('image', data=image_mm, compression="gzip")
        f.create_dataset('label', data=labels,   compression="gzip")
        f.close()

if __name__ == '__main__':
    if sys.argv[1] == 'h5':
        covert_h5(sys.argv[2])
    elif sys.argv[1] == 'label':
        label_stats(sys.argv[2])
    