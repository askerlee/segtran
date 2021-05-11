import os
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from common_util import get_filename
from dataloaders.datasets3d import make_brats_pred_consistent, brats_map_label, brats_inv_map_label, harden_segmap3d
from tqdm import tqdm
import pdb

# When test_interp is not None, it should be a tuple like (56,56,40) used as the size of interpolated masks.
def test_all_cases(net, db_test, task_name, net_type, num_classes, batch_size=8,
                   orig_patch_size=(128, 112, 80), input_patch_size=(128, 112, 64), 
                   stride_xy=56, stride_z=40, 
                   save_result=True, test_save_path=None, 
                   preproc_fn=None, test_interp=None, has_mask=True):
                    
    total_metric = np.zeros((num_classes - 1, 4))
    valid_metric_counts = np.zeros((num_classes - 1, 4))
    binarize = (num_classes == 2)
    
    for image_idx in tqdm(range(len(db_test))):
        sample = db_test[image_idx]
        # image_tensor is 4D. First dim is modality. 
        # If only one modality is chosen, then image_tensor.shape[0] = 1.
        image_tensor, mask_tensor = sample['image'].cuda(), sample['mask'].cuda()
        image_path = sample['image_path']
        image_name = get_filename(image_path)
        image_name = image_name.split(".")[0]
        
        if task_name == 'brats':
            # Map 4 to 3, and keep 0,1,2 unchanged.
            mask_tensor -= (mask_tensor == 4).long()
            mask_tensor = mask_tensor.float()
            mask_tensor = brats_map_label(mask_tensor, binarize)
        
        # preproc_fn is usually None for brats
        if preproc_fn is not None:
            image_tensor = preproc_fn(image_tensor)

        # mask:         (179, 147, 88),     uint8
        # mask_tensor:  (4, 179, 147, 88),  float32
        # preds_hard:   (179, 147, 88),     int64
        # prob_map:     (4, 179, 147, 88),  float32
        if test_interp is not None:
            if len(test_interp) == 1:
                interp_size = [ int(L*test_interp[0]) for L in mask_tensor.shape[1:] ]
            else:
                interp_size = [ int(L*test_interp[i]) for i, L in enumerate(mask_tensor.shape[1:]) ]
                    
            mask_tensor = mask_tensor.unsqueeze(dim=0)
            small_pred  = F.interpolate(mask_tensor, size=interp_size, mode='nearest')
            big_pred    = F.interpolate(small_pred, size=mask_tensor.shape[2:], mode='trilinear', align_corners=False)
            preds_soft  = big_pred[0]
            mask_tensor = mask_tensor[0]
            preds_soft  = make_brats_pred_consistent(preds_soft, is_conservative=False)
            preds_hard  = harden_segmap3d(preds_soft, 0.5)
        else:
            preds_hard, preds_soft = test_single_case(net, image_tensor, orig_patch_size, input_patch_size, 
                                                      batch_size, stride_xy, stride_z,
                                                      task_name, net_type, num_classes)
        
        if has_mask:    
            allcls_metric, allcls_metric_valid = calculate_metric_percase(preds_hard, mask_tensor, num_classes)
        else:
            allcls_metric, allcls_metric_valid = 0, 1
            
        print("%s:\n%s" %(image_path, allcls_metric))
        total_metric += allcls_metric
        valid_metric_counts += allcls_metric_valid
        
        if (image_idx+1) % 20 == 0:
            avg_metric = total_metric / valid_metric_counts
            print("{}:".format(image_idx+1))
            print(avg_metric)
            
        if save_result:
            if task_name == 'brats':
                inv_probs = brats_inv_map_label(preds_soft)
                preds_hard = torch.argmax(inv_probs, dim=0)
                # Map 3 to 4, and keep 0, 1, 2 unchanged.
                preds_hard += (preds_hard == 3).long()
            preds_hard_np = preds_hard.data.cpu().numpy()
            nib.save(nib.Nifti1Image(preds_hard_np.astype(np.float32), np.eye(4)), 
                     os.path.join(test_save_path, image_name + ".nii.gz"))
            
    avg_metric = total_metric / valid_metric_counts
    return avg_metric

def test_single_case(net, image, orig_patch_size, input_patch_size, batch_size, stride_xy, stride_z,
                     task_name, net_type, num_classes):
    C, H, W, D     = image.shape
    dx, dy, dz  = orig_patch_size
    
    # if any dimension of image is smaller than orig_patch_size, then padding it
    add_pad = False
    if H < dx:
        h_pad = dx - H
        add_pad = True
    else:
        h_pad = 0
    if W < dy:
        w_pad = dy - W
        add_pad = True
    else:
        w_pad = 0
    if D < dz:
        d_pad = dz - D
        add_pad = True
    else:
        d_pad = 0
        
    hl_pad, hr_pad = h_pad // 2, h_pad-h_pad // 2
    wl_pad, wr_pad = w_pad // 2, w_pad-w_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad-d_pad // 2
    if add_pad:
        image = F.pad(image, (0, 0, dl_pad, dr_pad, wl_pad, wr_pad, hl_pad, hr_pad), 
                      mode='constant', value=0)
    
    # New image dimensions after padding
    C, H2, W2, D2 = image.shape

    sx = math.ceil((H2 - dx) / stride_xy) + 1
    sy = math.ceil((W2 - dy) / stride_xy) + 1
    sz = math.ceil((D2 - dz) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    preds_soft  = torch.zeros((num_classes, ) + image.shape[1:], device='cuda')
    cnt         = torch.zeros_like(image[0])

    for x in range(0, sx):
        xs = min(stride_xy*x, H2-dx)
        yzs_batch = []
        test_patches = []
        
        for y in range(0, sy):
            ys = min(stride_xy * y, W2-dy)
            for z in range(0, sz):
                zs = min(stride_z * z, D2-dz)
                test_patch = image[:, xs:xs+dx, ys:ys+dy, zs:zs+dz]
                # test_patch: [1, 1, 144, 144, 80]
                test_patches.append(test_patch)
                yzs_batch.append([ys, zs])
                # When z == sz - 1, it's the last iteration.
                if len(test_patches) == batch_size or (y == sy - 1 and z == sz - 1):
                    # test_batch has a batch dimension after stack().
                    test_batch = torch.stack(test_patches, dim=0)
                    test_batch = F.interpolate(test_batch, size=input_patch_size,
                                               mode='trilinear', align_corners=False)
                    with torch.no_grad():
                        scores_raw = net(test_batch)
                    
                    if net_type == 'unet':
                        scores_raw = scores_raw[1]
                        
                    scores_raw = F.interpolate(scores_raw, size=orig_patch_size, 
                                               mode='trilinear', align_corners=False)
                    
                    probs = torch.sigmoid(scores_raw)
                    for i, (ys, zs) in enumerate(yzs_batch):
                        preds_soft[:, xs:xs+dx, ys:ys+dy, zs:zs+dz] += probs[i]
                        cnt[xs:xs+dx, ys:ys+dy, zs:zs+dz] += 1
                            
                    test_patches = []
                    yzs_batch = []

    preds_soft = preds_soft / cnt.unsqueeze(dim=0)
    if task_name == 'brats':
        preds_soft = make_brats_pred_consistent(preds_soft, is_conservative=False)
        preds_hard = torch.zeros_like(preds_soft)
        preds_hard[1:] = (preds_soft[1:] >= 0.5)
        # A voxel is background if it's not ET, WT, TC.
        preds_hard[0]  = (preds_hard[1:].sum(axis=0) == 0)
    else:
        # preds_hard: predicted mask; the mask with the highest predicted probabilities.
        preds_hard = torch.argmax(preds_soft, dim=0)

    if add_pad:
        # Remove padded pixels. clone() to make memory contiguous.
        preds_hard = preds_hard[:, hl_pad:hl_pad+H, wl_pad:wl_pad+W, dl_pad:dl_pad+D].clone()
        preds_soft = preds_soft[:, hl_pad:hl_pad+H, wl_pad:wl_pad+W, dl_pad:dl_pad+D].clone()
    return preds_hard, preds_soft

def calculate_metric_percase(allcls_pred, allcls_gt, num_classes):
    allcls_metric       = np.zeros((num_classes - 1, 4))
    allcls_metric_valid = np.ones((num_classes - 1, 4))
    allcls_pred_np  = allcls_pred.data.cpu().numpy()
    allcls_gt_np    = allcls_gt.data.cpu().numpy()
    
    for cls in range(1, num_classes):
        pred = allcls_pred_np[cls].astype(np.uint8)
        gt   = allcls_gt_np[cls].astype(np.uint8)
            
        dice = metric.binary.dc(pred, gt)
        if gt.sum() > 0:
            jc = metric.binary.jc(pred, gt)
        else:
            jc = 0
            allcls_metric_valid[cls-1, 1] = 0
            
        if pred.sum() > 0 and gt.sum() > 0:
            # hd = metric.binary.hd95(pred, gt)
            hd = 0
            asd  = metric.binary.asd(pred, gt)
        else:
            hd = 0
            asd = 0
            allcls_metric_valid[cls-1, 2] = 0
            allcls_metric_valid[cls-1, 3] = 0
            
        allcls_metric[cls-1] = [dice, jc, hd, asd]
        
    return allcls_metric, allcls_metric_valid
