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

def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, gt_mask_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the gt_mask.
    """
    assert input_logits.size() == gt_mask_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    gt_mask_softmax = F.softmax(gt_mask_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], gt_mask_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, gt_mask_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the gt_mask.
    """
    assert input_logits.size() == gt_mask_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    gt_mask_softmax = F.softmax(gt_mask_logits, dim=1)

    mse_loss = (input_softmax-gt_mask_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, gt_mask_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the gt_mask.
    """
    assert input_logits.size() == gt_mask_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    gt_mask_softmax = F.softmax(gt_mask_logits, dim=1)

    # return F.kl_div(input_log_softmax, gt_mask_softmax)
    kl_div = F.kl_div(input_log_softmax, gt_mask_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
