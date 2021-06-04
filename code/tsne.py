from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import sys
import itertools
import numpy as np
import random
import argparse
from internal_util import avg_hausdorff_np
np.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument("--pca", dest='pca_comp', type=int, default=-1,
                    help='Number of PCA components before t-SNE (Default: -1, no PCA).')
parser.add_argument('--n', dest='num_points_each_class', type=int,  default=1000, help='Number of points in each class')
parser.add_argument('--featcp', dest='feat_cp_filepaths', type=str, required=True,
                    help='Path(s) to the feature checkpoint file(s).')
parser.add_argument("--labelmode", dest='label_mode', type=str, default='cp-class',
                    choices=['cp-class', 'class', 'cp'],
                    help='The mode to assign labels across checkpoints')
parser.add_argument('--selclass', dest='selected_classes', type=str, default=None,
                    help='Selected classes to visualize in the cross cp mode')
parser.add_argument('--norm', dest='feature_norm_mode', type=str, default=None,
                    choices=[None, 'none', 'bn', 'layernorm'],
                    help='Which way to normalize features before doing t-SNE and calculating hausdorff distances')
parser.add_argument("--ds", dest='dataset_type', type=str, required=True,
                    choices=['fundus', 'polyp'],
                    help='Type of datasets')
parser.add_argument('--deterministic', type=int,  default=1, help='whether generate deterministic results')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

args = parser.parse_args()
args.feat_cp_filepaths = args.feat_cp_filepaths.split(",")
if args.selected_classes:
    args.selected_classes = [ int(c) for c in args.selected_classes.split(",") ]
if args.feature_norm_mode == 'none':
    args.feature_norm_mode = None
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("Set this session to deterministic mode")
    
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

cps_features = []
cps_classes  = []
cps_indices  = []
fundus_class_names = ['background', 'disc', 'cup']
polyp_class_names = ['background', 'polyp']
if args.dataset_type == 'fundus':
    class_names = fundus_class_names
if args.dataset_type == 'polyp':
    class_names = polyp_class_names
    
num_classes = len(class_names)
cp_names = []
cp_class_names = []

PALETTE = sns.color_palette('muted', n_colors=3)
CMAP    = ListedColormap(PALETTE.as_hex())
cp_markers = list("vx.")
num_cps = len(args.feat_cp_filepaths)
assert num_cps <= len(cp_markers)
feat_norm_layer = None

for cp_idx, feat_cp_filepath in enumerate(args.feat_cp_filepaths):
    features_dict = torch.load(feat_cp_filepath, map_location=torch.device('cpu'))
    features, labels = features_dict['features'], features_dict['labels']
    num_points, num_channels = features.shape
    print("{} {}-dim feature vectors loaded from '{}'".format(num_points, num_channels, feat_cp_filepath))

    N = args.num_points_each_class
    features_by_class = []
    labels_by_class = []

    for i in range(num_classes):
        cp_class_names.append("{}-{}".format(cp_idx, class_names[i]))
        if args.selected_classes and (i not in args.selected_classes):
            print("Skip class '{}'".format(class_names[i]))
            continue

        cls_features = features[ labels == i ]
        num_cls_points = len(cls_features)

        if num_cls_points > N:
            perm = torch.randperm(num_cls_points)
            chosen_indices = perm[:N]
            cls_features = cls_features[chosen_indices]
            print("'{}' has {} points. Choose {}".format(class_names[i], num_cls_points, N))

        features_by_class.append(cls_features)
        labels_by_class.append(torch.ones(len(cls_features), dtype=int) * i)

    cp_features = torch.cat(features_by_class)
    cp_classes = torch.cat(labels_by_class)
    cp_indices = torch.ones(len(cp_features), dtype=int) * cp_idx
    cps_features.append(cp_features)
    cps_classes.append(cp_classes)
    cps_indices.append(cp_indices)
    cp_names.append(str(cp_idx))

features = torch.cat(cps_features)
classes  = torch.cat(cps_classes)
cps_indices = torch.cat(cps_indices).numpy()

if not feat_norm_layer and args.feature_norm_mode:
    if args.feature_norm_mode == 'bn':
        feat_norm_layer = nn.BatchNorm1d(features.shape[-1])
        feat_norm_layer.train()   # Use batch stats to do normalization, instead of moving stats.
    elif args.feature_norm_mode == 'layernorm':
        feat_norm_layer = nn.LayerNorm(features.shape[-1])

print("Classes:", cp_class_names)
if args.feature_norm_mode:
    for i in range(num_classes):
        cls_selected = (classes == i)
        cls_features = features[cls_selected]
        with torch.no_grad():
            cls_features = feat_norm_layer(cls_features)
        features[cls_selected] = cls_features

features = features.numpy()
classes  = classes.numpy()

for i, cp_idx in itertools.product(range(num_classes), range(num_cps)):
    hausdorff_matrix = np.zeros((num_cps, num_classes))
    f1_selected = (classes == i) & (cps_indices == cp_idx)
    if f1_selected.sum() == 0:
        continue
    f1_subset = features[f1_selected]
    print("CP %d, class %d-%s" %(cp_idx, i, class_names[i]))
    
    for j, cp_idx2 in itertools.product(range(num_classes), range(num_cps)):
        f2_selected = (classes == j) & (cps_indices == cp_idx2)
        if f2_selected.sum() == 0:
            continue
        f2_subset = features[f2_selected]
        pair_hausdorff = avg_hausdorff_np(f1_subset, f2_subset)
        hausdorff_matrix[cp_idx2, j] = pair_hausdorff

    print(hausdorff_matrix)
    
if args.pca_comp > 0:
    pca = PCA(n_components=args.pca_comp)
    features_lowdim = pca.fit_transform(features)
else:
    features_lowdim = features
tsne = TSNE(n_components=2).fit_transform(features_lowdim)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
# class_colors = ['blue', 'red', 'green']

cp_class_idx = 0
for cp_idx in range(num_cps):
    # for every class, we'll add a scatter plot separately
    for i in range(num_classes):
        # extract the coordinates of the points of this cp and this class only
        points_selected = (classes == i) & (cps_indices == cp_idx)
        if points_selected.sum() == 0:
            cp_class_idx += 1
            continue
        current_tx = tx[points_selected]
        current_ty = ty[points_selected]

        if args.label_mode == 'cp-class':
            label = cp_class_names[cp_class_idx]
        elif args.label_mode == 'class':
            label = class_names[i]
        elif args.label_mode == 'cp':
            label = cp_names[cp_idx]
        else:
            breakpoint()

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=CMAP.colors[cp_idx], label=label, marker=cp_markers[i],
                   s=9, linewidth=0.7, alpha=0.8)
        cp_class_idx += 1

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()
