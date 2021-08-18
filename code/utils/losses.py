import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb

class SmoothDiceLoss(nn.Module):
    # Set momentum = 1 to disable smoothing.
    def __init__(self, momentum=0.1):
        super(SmoothDiceLoss, self).__init__()
        self.momentum = momentum
        self.running_denom = -1
        self.eps = 1e-5
        
    def forward(self, score, gt_mask):
        score = score.view(score.shape[0], -1)
        gt_mask = gt_mask.float().view(gt_mask.shape[0], -1)        
        intersect = torch.sum(score * gt_mask, dim=1)
        y_sum = torch.sum(gt_mask * gt_mask, dim=1)
        z_sum = torch.sum(score * score, dim=1)
        denom = z_sum + y_sum + self.eps
        mean_denom = denom.mean()
        
        if self.running_denom == -1:
            self.running_denom = mean_denom.item()
            dyn_offset = torch.zeros(1, device='cuda')
        else:
            # Update the running average of the denominator.
            self.running_denom = self.running_denom * (1 - self.momentum) \
                                 + mean_denom.item() * self.momentum
            # dyn_offset is a tensor of length nBatch.
            dyn_offset = self.running_denom - denom.data
        # Make denom + dyn_offset = self.running_denom.
        smooth_dice = (2 * intersect + self.eps + dyn_offset) / (denom + dyn_offset)
        smooth_loss = 1 - smooth_dice
        # odice: original dice. oloss: original loss.
        orig_dice = (2 * intersect + self.eps) / denom
        orig_loss = 1 - orig_dice
        
        smooth_loss = smooth_loss.mean()
        orig_loss   = orig_loss.mean()
        #print("r-denom: %.1f, denom: %s, offset: %s" % \
        #        (self.running_denom, str(denom.data.cpu().numpy()), str(dyn_offset.data.cpu().numpy())))
        return smooth_loss, orig_loss

# Compute each example's dice loss in a batch, and average them                
def dice_loss_indiv(score, gt_mask, weight=None):
    score = score.view(score.shape[0], -1)
    gt_mask = gt_mask.float().view(gt_mask.shape[0], -1)
    smooth = 1e-5
    intersect = torch.sum(score * gt_mask, dim=1)
    y_sum = torch.sum(gt_mask * gt_mask, dim=1)
    z_sum = torch.sum(score * score, dim=1)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - dice
    if weight is not None:
        loss = (loss * weight).mean()
    else:
        loss = loss.mean()
    return loss

# Treat the whole batch as one big example, and compute the dice loss.
def dice_loss_mix(score, gt_mask):
    gt_mask = gt_mask.float()
    smooth = 1e-5
    intersect = torch.sum(score * gt_mask)
    y_sum = torch.sum(gt_mask)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

# cd_area_ratio: cup/disc area ratio (differentiable).
# vcdr: vertical cup/disc ratio (non-differentiable).
def calc_cd_area_ratio(mask_nhot_soft, thres=0.5, calc_vcdr=False):
    # mask_nhot: 0: background. 1: disc. 2: cup.
    mask_nhot = (mask_nhot_soft >= thres)
    # Has the batch dimension.
    # The returned cd_area_ratio, vcdr are vectors (a batch of ratios).
    if mask_nhot.ndim == 4:
        disc_area   = mask_nhot[:, 1].sum(dim=2).sum(dim=1)
        cup_area    = mask_nhot[:, 2].sum(dim=2).sum(dim=1)
        cd_area_ratio   = cup_area / (disc_area + 0.0001)
        if calc_vcdr:
            # vert_occupied: [B, H]
            disc_vert_occupied  = (mask_nhot[:, 1].sum(dim=2) > 0)
            disc_vert_len       = disc_vert_occupied.argmax(dim=1) \
                                  - disc_vert_occupied.argmin(dim=1) + 1
            cup_vert_occupied   = (mask_nhot[:, 2].sum(dim=2) > 0)
            cup_vert_len        = cup_vert_occupied.argmax(dim=1)  \
                                  - cup_vert_occupied.argmin(dim=1)  + 1
            vcdr = cup_vert_len / (disc_vert_len + 0.0001)
        
            return cd_area_ratio, vcdr
        else:
            return cd_area_ratio
    # No batch dimension.
    # The returned cd_area_ratio, vcdr are scalars.
    else:
        disc_area   = mask_nhot[1].sum()
        cup_area    = mask_nhot[2].sum()
        cd_area_ratio   = cup_area / (disc_area + 0.0001)
        if calc_vcdr:
            # vert_occupied: [B, H]
            disc_vert_occupied  = (mask_nhot[1].sum(dim=1) > 0)
            disc_vert_len       = disc_vert_occupied.argmax(dim=1) \
                                  - disc_vert_occupied.argmin(dim=1) + 1
            cup_vert_occupied   = (mask_nhot[2].sum(dim=1) > 0)
            cup_vert_len        = cup_vert_occupied.argmax(dim=1)  \
                                  - cup_vert_occupied.argmin(dim=1)  + 1
            vcdr = cup_vert_len / (disc_vert_len + 0.0001)
        
            return cd_area_ratio, vcdr
        else:
            return cd_area_ratio
                    
