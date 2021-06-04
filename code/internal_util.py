import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import collections
from matplotlib import cm
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
import receptivefield
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import cv2
from train_util import remove_norms, batch_norm
from test_util2d import calc_batch_metric
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

def visualize_model(net, vis_mode, vis_layers, patch_size, dataset=None):
    remove_norms(net, 'net')
    input_shape = list(patch_size) + [3]
    rf = PytorchReceptiveField(lambda: net)
    if dataset is None:
        probe_image = get_default_image(input_shape, name='cat')
        probe_mask  = None
    else:
        probe_image_dict = dataset[24]
        probe_image = probe_image_dict['image']
        probe_mask  = probe_image_dict['mask']
        image_path  = probe_image_dict['image_path']
        print(image_path)
        # probe_image: np.array (576, 576, 3)
        # probe_mask:  np.array (3, 576, 576)
        probe_image = cv2.resize(probe_image, patch_size)

    rf_params = rf.compute(input_shape = input_shape)

    '''
    # plot receptive fields
    rf.plot_rf_grids(
        custom_image=cat_image,
        figsize=(20, 12),
        layout=(1, 3))
    '''

    # visualize all layers
    if vis_layers is None:
        vis_layers = list(range(net.num_vis_layers))

    for i in vis_layers:
        feat_size = net.feature_maps[i].shape[2:]
        center = (min(feat_size[0]//2, 128), min(feat_size[1]//2, 128))
        print(net.feature_maps[i].shape, center)
        rf.plot_gradient_at(fm_id=i, point=center, image=probe_image, figsize=(7, 7))

    plt.show()

# Revised from https://github.com/pytorch/pytorch/issues/35642
class CachedDataset(Dataset):
    def __init__(self, dataset):
        super(CachedDataset, self).__init__()
        self.cache = dict()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
                
    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]
        sample = self.dataset[index]
        self.cache[index] = sample
        return sample

def pearson(t1, t2, dim=-1):
    if t1.shape != t2.shape:
        breakpoint()

    if dim == -1:
        t1flat = t1.flatten()
        t2flat = t2.flatten()
        t1flatz = t1flat - t1flat.mean()
        t2flatz = t2flat - t2flat.mean()
        norm1 = (t1flatz**2).float().sum().sqrt()
        norm2 = (t2flatz**2).float().sum().sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1

        corr = (t1flatz * t2flatz).float().sum() / (norm1 * norm2)
        return corr.item()

# pearson between the left/right halves of the tensor.
def lr_pearson(t1):
    left, right = t1.chunk(2, dim=-1)
    return pearson(left, right)

def initialize_reference_features(ref_feat_cp_path, num_ref_features, num_classes, selected_ref_classes, random_seed):
    features_dict = torch.load(ref_feat_cp_path, map_location=torch.device('cuda'))
    features, labels = features_dict['features'], features_dict['labels']
    num_points, num_channels = features.shape
    print("{} {}-dim reference feature vectors loaded from '{}'".format(num_points, num_channels, ref_feat_cp_path))

    N = num_ref_features
    ref_features_by_class = []
    
    for i in range(num_classes):
        if selected_ref_classes and (i not in selected_ref_classes):
            print("Skip class {}".format(i))
            ref_features_by_class.append(None)
            continue

        cls_features = features[ labels == i ]
        num_cls_points = len(cls_features)

        if num_cls_points > N:
            perm = torch.randperm(num_cls_points)
            chosen_indices = perm[:N]
            print(chosen_indices.sum())
            cls_features = cls_features[chosen_indices]
            print("Class {} has {} points. Choose {}".format(i, num_cls_points, N))
        
        cls_features.requires_grad = False
        ref_features_by_class.append(cls_features)

    return ref_features_by_class
        
def avg_hausdorff_np(A, B, exclude_id=True):
    AB_dists = cdist(A, B)
    if exclude_id:
        AB_dists[AB_dists==0] = 1000000
    A_dists = AB_dists.min(axis=1)
    B_dists = AB_dists.min(axis=0)
    avg_A_dist = A_dists.mean()
    avg_B_dist = B_dists.mean()
    sym_dist = (avg_A_dist + avg_B_dist) / 2
    return sym_dist

def avg_hausdorff(A, B, norm_p=2, topk=1, one_way=False, exclude_id=True):
    AB_dists = torch.cdist(A, B, p=norm_p)
    if exclude_id:
        AB_dists[AB_dists==0] = 1000000
    A_dists = AB_dists.topk(topk, largest=False, dim=1)[0]
    avg_A_dist = A_dists.mean()
    
    if one_way:
        return avg_A_dist
    else:
        B_dists = AB_dists.min(dim=0)[0]
        avg_B_dist = B_dists.mean()
        sym_dist = (avg_A_dist + avg_B_dist) / 2
        return sym_dist

def calc_contrast_losses(args, features, exclusive_mask_batch, ref_features_by_class, class_weights):
    total_pos_contrast_loss, total_neg_contrast_loss = 0, 0
    mask_batch_small = F.interpolate(exclusive_mask_batch, size=features.shape[2:], 
                                     mode='bilinear', align_corners=False)
    onehot_labels = (mask_batch_small >= 0.5)

    for cls in range(1, args.num_classes):                
        if ref_features_by_class[cls] is None:
            continue
        cls_features = features.transpose(1,3)[ onehot_labels[:, cls] ]
        if args.num_contrast_features > 0 and len(cls_features) > args.num_contrast_features:
            perm = torch.randperm(len(cls_features))
            chosen_indices = perm[:args.num_contrast_features]
            cls_features   = cls_features[chosen_indices]
        
        pos_ref_features = ref_features_by_class[cls]
        if len(pos_ref_features) > args.num_ref_features:
            perm = torch.randperm(len(pos_ref_features))
            chosen_indices      = perm[:args.num_ref_features]
            pos_ref_features    = pos_ref_features[chosen_indices]
    
        pos_contrast_loss = avg_hausdorff(cls_features, pos_ref_features, norm_p=2, topk=3,
                                          one_way=True, exclude_id=False)
        total_pos_contrast_loss += pos_contrast_loss * class_weights[cls]

        if args.do_neg_contrast:
            neg_cls = (cls + np.random.randint(1, args.num_classes)) % args.num_classes
            assert neg_cls != cls
            neg_ref_features = ref_features_by_class[neg_cls]
            if len(neg_ref_features) > args.num_ref_features:
                perm = torch.randperm(len(neg_ref_features))
                chosen_indices = perm[:args.num_ref_features]
                neg_ref_features   = neg_ref_features[chosen_indices]
    
            neg_contrast_loss = avg_hausdorff(cls_features, neg_ref_features, norm_p=2, topk=3,
                                              one_way=True, exclude_id=False)

            total_neg_contrast_loss += 0.5 * neg_contrast_loss * class_weights[cls]
                        
    return total_pos_contrast_loss, total_neg_contrast_loss
    
# aug_degrees: (aug_min_degree, aug_max_degree). The bigger, the higher degree of aug is applied.
def eval_robustness(args, net, refnet, dataloader, mask_prepred_mapping_func=None):
    AUG_DEG = args.robust_aug_degrees
    if not isinstance(AUG_DEG, collections.abc.Iterable):
        AUG_DEG = (AUG_DEG, AUG_DEG)

    if args.robustness_augs:
        augs = [ args.robustness_augs ]
        is_resize = [ False ]
    else:
        augs = [
            transforms.ColorJitter(brightness=AUG_DEG),
            transforms.ColorJitter(contrast=AUG_DEG),
            transforms.ColorJitter(saturation=AUG_DEG),
            transforms.Resize((192, 192)),
            transforms.Resize((432, 432)),
            transforms.Pad(0)   # Placeholder. Replace input images with random noises.
        ]
        is_resize = [ False, False, False, True, True, False ]

    num_augs = len(augs)
    num_iters = args.robust_sample_num // args.batch_size
    # on_pearsons: pearsons between old and new feature maps
    on_pearsons = np.zeros((num_augs, net.num_vis_layers))
    # lr_old_pearsons/lr_new_pearsons: pearsons between left-half and right-half of the feature maps
    lr_old_pearsons = np.zeros((net.num_vis_layers))
    old_stds        = np.zeros((net.num_vis_layers))
    lr_new_pearsons = np.zeros((num_augs, net.num_vis_layers))
    new_stds        = np.zeros((num_augs, net.num_vis_layers))
    aug_counts      = np.zeros(num_augs) + 0.0001
    print("Evaluating %d augs on %d layers of feature maps, %d samples" %(num_augs, net.num_vis_layers, args.robust_sample_num))
    do_BN = True
    orig_allcls_dice_sum    = np.zeros(args.num_classes - 1)
    aug_allcls_dice_sum     = np.zeros((num_augs, args.num_classes - 1))
    orig_sample_count       = 0
    aug_sample_counts       = np.zeros(num_augs) + 0.0001

    # Compare the feature maps from the same network.
    if refnet is None:
        refnet = net
        
    for it in tqdm(range(num_iters)):
        aug_idx = it % num_augs
        aug_counts[aug_idx] += 1
        aug = augs[aug_idx]
        dataloader.dataset.image_trans_func2 = transforms.Compose( [ aug ] + \
                                                                   dataloader.dataset.image_trans_func.transforms )

        batch = next(iter(dataloader))
        image_batch, image2_batch, mask_batch = batch['image'].cuda(), batch['image2'].cuda(), batch['mask'].cuda()
        image_batch = F.interpolate(image_batch, size=args.patch_size,
                                   mode='bilinear', align_corners=False)
        image2_batch = F.interpolate(image2_batch, size=args.patch_size,
                                   mode='bilinear', align_corners=False)
        if mask_prepred_mapping_func:
            mask_batch = mask_prepred_mapping_func(mask_batch)

        orig_input_size = mask_batch.shape[2:]
        if it == 0:
            print("Input size: {}, orig image size: {}".format(image_batch.shape[2:], orig_input_size))

        if aug_idx == 5:
            image2_batch.normal_()

        with torch.no_grad():
            scores_raw = refnet(image_batch)
            scores_raw = F.interpolate(scores_raw, size=orig_input_size,
                                       mode='bilinear', align_corners=False)

            batch_allcls_dice = calc_batch_metric(scores_raw, mask_batch, args.num_classes, 0.5)
            orig_allcls_dice_sum    += batch_allcls_dice.sum(axis=0)
            orig_sample_count       += len(batch_allcls_dice)

            orig_features = copy.copy(refnet.feature_maps)
            orig_bn_features = list(orig_features)
            net.feature_maps = []
            scores_raw2 = net(image2_batch)

            batch_allcls_dice = calc_batch_metric(scores_raw2, mask_batch, args.num_classes, 0.5)
            aug_allcls_dice_sum[aug_idx]    += batch_allcls_dice.sum(axis=0)
            aug_sample_counts[aug_idx]      += len(batch_allcls_dice)

            aug_features  = copy.copy(net.feature_maps)
            aug_bn_features  = list(aug_features)
            net.feature_maps = []
            for layer_idx in range(net.num_vis_layers):
                if is_resize[aug_idx] and orig_features[layer_idx].shape != aug_features[layer_idx].shape:
                    try:
                        aug_features[layer_idx] = F.interpolate(aug_features[layer_idx], size=orig_features[layer_idx].shape[-2:],
                                                                mode='bilinear', align_corners=False)
                    except:
                        breakpoint()

                if do_BN and orig_features[layer_idx].dim() == 4:
                    orig_bn_features[layer_idx] = batch_norm(orig_features[layer_idx])
                    aug_bn_features[layer_idx]  = batch_norm(aug_features[layer_idx])

                pear = pearson(orig_bn_features[layer_idx], aug_bn_features[layer_idx])
                on_pearsons[aug_idx, layer_idx]     += pear
                lr_old_pearsons[layer_idx] += lr_pearson(orig_bn_features[layer_idx])
                lr_new_pearsons[aug_idx, layer_idx] += lr_pearson(aug_bn_features[layer_idx])

                # 4D feature maps. Assume dim 1 is the channel dim.
                if orig_features[layer_idx].dim() == 4:
                    chan_num = orig_features[layer_idx].shape[1]
                    old_std  = orig_features[layer_idx].transpose(0, 1).reshape(chan_num, -1).std(dim=1).mean()
                    new_std  = aug_features[layer_idx].transpose(0, 1).reshape(chan_num, -1).std(dim=1).mean()
                else:
                    old_std  = orig_features[layer_idx].std()
                    new_std  = aug_features[layer_idx].std()
                old_stds[layer_idx] += old_std
                new_stds[aug_idx, layer_idx] += new_std

    aug_counts = np.expand_dims(aug_counts, 1)
    aug_sample_counts = np.expand_dims(aug_sample_counts, 1)
    on_pearsons /= aug_counts
    lr_old_pearsons /= num_iters
    lr_new_pearsons /= aug_counts
    old_stds /= num_iters
    new_stds /= aug_counts

    orig_allcls_avg_metric = orig_allcls_dice_sum / orig_sample_count
    aug_allcls_avg_metric  = aug_allcls_dice_sum  / aug_sample_counts

    print('Orig dices: ', end='')
    for cls in range(1, args.num_classes):
        orig_dice = orig_allcls_avg_metric[cls-1]
        print('%.3f ' %(orig_dice), end='')
    print()

    for aug_idx in range(num_augs):
        print('Aug %d dices: ' %aug_idx, end='')
        for cls in range(1, args.num_classes):
            aug_dice = aug_allcls_avg_metric[aug_idx, cls-1]
            print('%.3f ' %(aug_dice), end='')
        print()

    for layer_idx in range(net.num_vis_layers):
        print("%d: Orig LR P %.3f, Std %.3f" %(layer_idx, lr_old_pearsons[layer_idx], old_stds[layer_idx]))

    for aug_idx in range(num_augs):
        print(augs[aug_idx])
        for layer_idx in range(net.num_vis_layers):
            print("%d: ON P %.3f, LR P %.3f, Std %.3f" %(layer_idx,
                            on_pearsons[aug_idx, layer_idx],        # old/new    pearson
                            lr_new_pearsons[aug_idx, layer_idx],    # left/right pearson
                            new_stds[aug_idx, layer_idx]))
