import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
import sys
import pdb

output_size =[112, 112, 80]

def covert_h5(root, cutoff=0):
    do_localization = True
    if 'validation' in root.lower():
        is_training = False
    else:
        is_training = True

    listt = glob( os.path.join(root, '*/lgemri.nrrd') )
    for image_filename in tqdm(listt):
        image, img_header = nrrd.read(image_filename)
        labels, gt_header = nrrd.read(image_filename.replace('lgemri.nrrd', 'laendo.nrrd'))
        image_shape = image.shape
        H, W, D = labels.shape
        image = image.astype(np.float32)
        labels = (labels == 255).astype(np.uint8)
        
        if is_training and do_localization:
            tempL = np.nonzero(labels)
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])

            px = max(output_size[0] - (maxx - minx), 0) // 2
            py = max(output_size[1] - (maxy - miny), 0) // 2
            pz = max(output_size[2] - (maxz - minz), 0) // 2
            minx = max(minx - np.random.randint(10, 20) - px, 0)
            maxx = min(maxx + np.random.randint(10, 20) + px, H)
            miny = max(miny - np.random.randint(10, 20) - py, 0)
            maxy = min(maxy + np.random.randint(10, 20) + py, W)
            minz = max(minz - np.random.randint(5,  10) - pz, 0)
            maxz = min(maxz + np.random.randint(5,  10) + pz, D)
        elif is_training:
            tempL = np.nonzero(image > cutoff)
            # Find the boundary of non-zero voxels
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        # Do not crop black margins of validation images.
        else:
            minx = miny = minz = 0
            maxx, maxy, maxz = H, W, D

        nonzero_mask = (image > cutoff)
        N = np.sum(nonzero_mask)
        # The mean and std are calculated on nonzero voxels only.
        mean = np.sum(image) / N
        var  = np.sum(image * image) / N - mean * mean
        std  = np.sqrt(var)
        image = (image - mean) / std
        
        # Set voxels back to 0 if they are 0 before normalization.
        image *= nonzero_mask
        image  = image[minx:maxx,  miny:maxy]
        labels = labels[minx:maxx, miny:maxy]
        print("\n%s: %s => %s, %s" %(image_filename, image_shape, image.shape, labels.shape))
            
        print(labels.shape)
        f = h5py.File(image_filename.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image,  compression="gzip")
        f.create_dataset('label', data=labels, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5(sys.argv[1])
    