import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import nibabel as nib
import sys
import os

output_size =[128, 128, 80]

def covert_h5(root):
    listt = glob( os.path.join(root, 'imagesTr/*.nii.gz') )
    do_localization = False
    
    for item in tqdm(listt):
        image_obj = nib.load(item)
        image = image_obj.get_fdata()
        label_obj = nib.load(item.replace('imagesTr', 'labelsTr'))
        label = label_obj.get_fdata()
        label = label.astype(np.uint8)
        # label = (label >= 1).astype(np.uint8)
        w, h, d = label.shape
        image_shape = image.shape

        if do_localization:
            tempL = np.nonzero(label)
            # Find the boundary of non-zero labels
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])

            # px, py, pz ensure the output image is at least of output_size
            px = max(output_size[0] - (maxx - minx), 0) // 2
            py = max(output_size[1] - (maxy - miny), 0) // 2
            pz = max(output_size[2] - (maxz - minz), 0) // 2
            # randint(10, 20) lets randomly-sized zero margins included in the output image
            minx = max(minx - np.random.randint(10, 20) - px, 0)
            maxx = min(maxx + np.random.randint(10, 20) + px, w)
            miny = max(miny - np.random.randint(10, 20) - py, 0)
            maxy = min(maxy + np.random.randint(10, 20) + py, h)
            minz = max(minz - np.random.randint(5, 10) - pz, 0)
            maxz = min(maxz + np.random.randint(5, 10) + pz, d)
        else:
            tempL = np.nonzero(image)
            # Find the boundary of non-zero labels
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])
                        
        image = image[minx:maxx, miny:maxy, minz:maxz]
        image = image.astype(np.float32)
        if len(image.shape) == 4:
            MOD = image.shape[3]
            for m in range(MOD):
                image[:, :, :, m] = (image[:, :, :, m] - np.mean(image[:, :, :, m])) / np.std(image[:, :, :, m])
        else:
            image = (image - np.mean(image)) / np.std(image)
            
        label = label[minx:maxx, miny:maxy, minz:maxz]
        print("%s: %s => %s, %s" %(item, image_shape, image.shape, label.shape))

        f = h5py.File(item.replace('.nii.gz', '.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5(sys.argv[1])
    