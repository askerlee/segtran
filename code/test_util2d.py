import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import cv2
from common_util import get_filename
from dataloaders.datasets2d import harden_segmap2d, load_mask, onehot_inv_map
from torchvision.transforms import ToTensor
from utils.losses import calc_vcdr
import pdb

# If don't want to save test predictions, set test_save_path=None.            
# When test_interp is not None, it should be a tuple like (144,144) used as the size of interpolated masks.
# We assume the model always do pred_sep_classes, i.e., each pixel's N candidate classes 
# are independently predicted as N binary variables.
# prepred: pre-prediction. postpred: post-prediction.
# If reload_mask, read raw masks from original input files. This is used when input images are in varying sizes.
def test_all_cases(net, dataloader, task_name, num_classes, mask_thres, model_type,
                   orig_input_size, patch_size, stride, 
                   test_save_paths=None, out_origsize=False, 
                   mask_prepred_mapping_func=None, mask_postpred_mapping_funcs=None,
                   reload_mask=False, test_interp=None, do_calc_vcdr_error=False, 
                   save_features_img_count=0, save_features_file_path=None,
                   save_ext='png', verbose=False):
    allcls_metric_sum   = np.zeros(num_classes - 1 + do_calc_vcdr_error)
    allcls_metric_count = np.zeros(num_classes - 1 + do_calc_vcdr_error)
    features_img_saved_count = 0
    saved_features = []
    saved_labels   = []
    
    for i_batch, data_batch in tqdm(enumerate(dataloader)):
        image_batch, mask_batch       = data_batch['image'], data_batch['mask']
        image_batch, mask_batch       = image_batch.cuda(),  mask_batch.cuda()
        image_paths, mask_paths       = data_batch['image_path'], data_batch['mask_path']
        crop_pos_batch = data_batch['crop_pos']
        unscaled_sizes, uncropped_sizes = data_batch['unscaled_size'], data_batch['uncropped_size']
                
        if reload_mask:
            orig_mask_batch = []
            for mask_path in mask_paths:
                mask = load_mask(mask_path, binarize=(num_classes==2))
                mask = ToTensor()(mask).cuda()
                if mask_prepred_mapping_func:
                    mask = mask_prepred_mapping_func(mask)
                orig_mask_batch.append(mask)
            orig_mask_batch = torch.stack(orig_mask_batch)
        else:
            orig_mask_batch = mask_batch
            
        if mask_prepred_mapping_func:
            orig_mask_batch = mask_prepred_mapping_func(orig_mask_batch)
                
        # predictions:  (B, 3, 256, 256), int64
        # preds_soft:     (B, 3, 256, 256), float32
        # mask_batch: (B, 3, 256, 256), float32

        if test_interp is not None:
            small_pred  = F.interpolate(mask_batch, size=test_interp, mode='nearest')
            big_pred    = F.interpolate(small_pred, size=mask_batch.shape[2:], mode='bilinear', align_corners=False)
            preds_soft  = big_pred
            preds_hard  = harden_segmap2d(big_pred, 0.5)
        else:
            preds_hard, preds_soft = test_single_batch(net, image_batch, orig_input_size, 
                                                       patch_size, stride, task_name, 
                                                       num_classes, mask_thres, model_type)
        
        batch_metric = calc_batch_metric(preds_soft, orig_mask_batch, num_classes, mask_thres, do_calc_vcdr_error=do_calc_vcdr_error)

        if verbose:
            print("%s... (%d images):\n%s" %(get_filename(image_paths[0]), len(image_batch), batch_metric))
            
        allcls_metric_sum   += batch_metric.sum(axis=0)
        allcls_metric_count += len(batch_metric)
        
        if features_img_saved_count < save_features_img_count:
            features = net.feature_maps[-1]
            exclusive_mask_batch = mask_prepred_mapping_func(mask_batch, exclusive=True)
            mask_batch_small = F.interpolate(exclusive_mask_batch, size=features.shape[2:], 
                                             mode='bilinear', align_corners=False)
            onehot_labels = (mask_batch_small >= 0.5).float()
            # onehot_inv_map() duplicates the channel by 3 times. Only 1 is necessary.
            labels = onehot_inv_map(onehot_labels)[:, :, :, 0]
            saved_features.append(features.transpose(1, -1).reshape(-1, features.shape[1]))
            saved_labels.append(labels.flatten())
            features_img_saved_count += len(features)
            
        if test_save_paths is not None:
            test_save_path_soft, test_save_path_hard = test_save_paths
            
            for i, image_path in enumerate(image_paths):
                crop_x, crop_y  = crop_pos_batch[i].numpy()
                H0, W0          = unscaled_sizes[i].numpy()
                unscaled_pred_soft  = F.interpolate(preds_soft[i].unsqueeze(0), 
                                                    size=(H0, W0), mode='bilinear', 
                                                    align_corners=False)
                unscaled_pred_soft  = unscaled_pred_soft[0]
                unscaled_pred_hard  = harden_segmap2d(unscaled_pred_soft, mask_thres)
                unscaled_pred_soft  = unscaled_pred_soft.permute([1, 2, 0])
                # fundus_inv_map_mask should be in mask_postpred_mapping_funcs.
                # It maps 3-channel 0/1 seg scores to 1-channel 0~255 pixel values.
                if mask_postpred_mapping_funcs:
                    for func in mask_postpred_mapping_funcs:
                        unscaled_pred_hard = func(unscaled_pred_hard)
                
                pred_zip = zip(('soft', 'hard'), (unscaled_pred_soft, unscaled_pred_hard), test_save_paths)
                
                for pred_type, unscaled_pred, test_save_path in pred_zip:
                    unscaled_pred_np = unscaled_pred.data.cpu().numpy()
                    if pred_type == 'soft':
                        # unable to save soft masks for number of classes other than 3.
                        if num_classes != 3:
                            continue
                        unscaled_pred_np = (unscaled_pred_np * 255).astype(np.uint8)
                    # Otherwise, fundus_inv_map_mask has already mapped the 
                    # 3-channel 0/1 hard seg scores to 1-channel 0/128/255 pixel values.
                    # So no matter what out_softscores is, unscaled_pred_np already contains legit pixel values.
                    
                    if out_origsize:
                        uncropped_size  = tuple(uncropped_sizes[i].data.cpu().numpy())
                        if pred_type == 'soft':
                            uncropped_pred_np = np.zeros(uncropped_size + (unscaled_pred_np.shape[2],), dtype=np.uint8)
                            # By default, predict all pixels as background.
                            uncropped_pred_np[:, :, 0] = 255
                            uncropped_pred_np[crop_x:crop_x+H0, crop_y:crop_y+W0] = unscaled_pred_np
                        else:
                            uncropped_pred_np = np.ones(uncropped_size, dtype=np.uint8) * 255
                            uncropped_pred_np[crop_x:crop_x+H0, crop_y:crop_y+W0] = unscaled_pred_np
                                
                        pred_seg_img = Image.fromarray(uncropped_pred_np)
                    else:
                        pred_seg_img = Image.fromarray(unscaled_pred_np)
                        
                    image_name = get_filename(image_path)
                    image_trunk = image_name.split(".")[0].split("_")[0]
                    image_name2 = "{}.{}".format(image_trunk, save_ext)
                    pred_seg_img.save(os.path.join(test_save_path, image_name2))
                
            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            # nib.save(nib.Nifti1Image(mask[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

    allcls_avg_metric = allcls_metric_sum / allcls_metric_count
    if features_img_saved_count > 0:
        saved_features = torch.cat(saved_features, dim=0)
        saved_labels   = torch.cat(saved_labels)
        torch.save({ 'features': saved_features, 'labels': saved_labels }, save_features_file_path)
        print("{} feature vectors saved to '{}'".format(len(saved_features), save_features_file_path))
        
    return allcls_avg_metric, allcls_metric_count

def test_single_batch(net, image_batch, orig_input_size, patch_size, 
                      stride, task_name, num_classes, mask_thres, model_type):
    B, C, H, W = image_batch.shape
    dx, dy = orig_input_size
    # if any dimension of image is smaller than orig_input_size, then padding it
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
        
    hl_pad, hr_pad = h_pad // 2, h_pad-h_pad // 2
    wl_pad, wr_pad = w_pad // 2, w_pad-w_pad // 2
    if add_pad:
        image_batch = F.pad(image_batch, (wl_pad, wr_pad, hl_pad, hr_pad), 
                            mode='constant', value=0)
    # New image dimensions after padding
    H2, W2 = image_batch.shape[2:]

    sx = math.ceil((H2 - dx) / stride[0]) + 1
    sy = math.ceil((W2 - dy) / stride[1]) + 1
    # print("{}, {}, {}".format(sx, sy))
    preds_shape = list(image_batch.shape)
    preds_shape[1] = num_classes
    preds_soft = torch.zeros(preds_shape, device='cuda')
    cnt = torch.zeros_like(preds_soft[:, 0])
    
    for x in range(0, sx):
        xs = min(stride[0]*x, H2-dx)
        for y in range(0, sy):
            ys = min(stride[1] * y, W2-dy)
            # test_patch: [10, 3, 576, 576]
            test_patch = image_batch[:, :, xs:xs+dx, ys:ys+dy]
            test_patch = F.interpolate(test_patch, size=patch_size,
                                       mode='bilinear', align_corners=False)
            # test_patch: [10, 3, 288, 288]
            with torch.no_grad():
                scores_raw = net(test_patch)
            if model_type == 'pranet':
                # Use lateral_map_2 for single-loss training.
                # Outputs is missing one channel (background). 
                # As the background doesn't incur any loss, its value doesn't matter. 
                # So add an all-zero channel to it.
                scores_raw0 = scores_raw[3]
                background = torch.zeros_like(scores_raw0[:, [0]])
                scores_raw = torch.cat([background, scores_raw0], dim=1)
            
            if model_type == 'nnunet':
                scores_raw = scores_raw[0]
                
            scores_raw = F.interpolate(scores_raw, size=orig_input_size, 
                                       mode='bilinear', align_corners=False)
                                    
            probs = torch.sigmoid(scores_raw)
            preds_soft[:, :, xs:xs+dx, ys:ys+dy] += probs
            cnt[:, xs:xs+dx, ys:ys+dy] += 1
                                                        
    preds_soft = preds_soft / cnt.unsqueeze(dim=1)
    preds_hard = harden_segmap2d(preds_soft, mask_thres)

    if add_pad:
        preds_hard = preds_hard[:, :, hl_pad:hl_pad+H, wl_pad:wl_pad+W]
        preds_soft = preds_soft[:, :, hl_pad:hl_pad+H, wl_pad:wl_pad+W]

    return preds_hard, preds_soft

# Assume two types of input:
# 1) predictions, gt_mask are both 3D tensors: (batch, H, W), In this case, return a batch of dice scores.
# 2) both are 2D tensors: (H, W). In this case, return a single dice score.
# Arbitrary zero-padding doesn't change the dice scores.
def calc_dice(predictions, gt_mask):
    gt_mask = gt_mask.float()
    smooth = 1e-5
    intersect = torch.sum(predictions * gt_mask, dim=(-1,-2))
    y_sum = torch.sum(gt_mask * gt_mask, dim=(-1,-2))
    z_sum = torch.sum(predictions * predictions, dim=(-1,-2))
    batch_dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return batch_dice

# Do not assume instances in a batch have the same shape.
# So BC_pred, BC_gt are two lists of 3-D tensors (class, H, W), 
# instead of two 4-D tensors (batch, class, H, W).  
def calc_batch_metric(BC_pred_soft, BC_gt, num_classes, mask_thres, do_calc_vcdr_error=False):
    batch_size  = len(BC_pred_soft)
    batch_allcls_dice = np.zeros((batch_size, num_classes - 1 + do_calc_vcdr_error))
    
    for ins in range(0, batch_size):
        C_pred_soft = BC_pred_soft[ins]
        C_gt        = BC_gt[ins]
        # Resize to original mask size. To deal with varying input image sizes.
        C_pred_soft = F.interpolate(C_pred_soft.unsqueeze(0), size=C_gt.shape[1:], mode='bilinear', align_corners=False)
        C_pred_soft = C_pred_soft[0]
        C_pred = harden_segmap2d(C_pred_soft, mask_thres)
            
        for cls in range(1, num_classes):
            pred    = C_pred[cls]
            gt_mask = C_gt[cls]
            batch_dice = calc_dice(pred, gt_mask)
            batch_allcls_dice[ins, cls-1] = batch_dice.cpu().numpy()
        
        if do_calc_vcdr_error:
            vcdr_gt     = calc_vcdr(C_gt)
            vcdr_pred   = calc_vcdr(C_pred)
            vcdr_error  = np.abs((vcdr_gt - vcdr_pred).cpu().numpy())
            batch_allcls_dice[ins, num_classes - 1] = vcdr_error
            
    return batch_allcls_dice

def remove_fragmentary_segs(segmap, bg_value):
    if type(segmap) == torch.Tensor:
        segmap_np = segmap.data.cpu().numpy()
    else:
        segmap_np = segmap
        
    bgfg_segmap = (segmap_np != bg_value).astype(np.uint8)
    comp_num, mask_comp = cv2.connectedComponents(bgfg_segmap)
    values, counts = np.unique(mask_comp, return_counts=True)
    # Find the top-2 frequent values
    try:
        v1, v2 = np.argpartition(counts, -2)[-2:]
    except:
        pdb.set_trace()
    frag_mask = np.logical_and(mask_comp != v1, mask_comp != v2)
    segmap_np[frag_mask] = bg_value
    
    if type(segmap) == torch.Tensor:
        segmap2 = torch.tensor(segmap_np, device=segmap.device)
    else:
        segmap2 = segmap_np
        
    return segmap2
    