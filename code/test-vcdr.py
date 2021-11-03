from dataloaders.datasets2d import fundus_map_mask
from utils.losses import calc_vcdr
from PIL import Image
from os import listdir
from os.path import join
import sys
import numpy as np
import torch
import argparse
from vcdr_acc import calc_vcdr_acc

parser = argparse.ArgumentParser()
# maskdir of the predicted segmentation masks
parser.add_argument('--maskdir', type=str)
# delta used to fix the disc/cup vertical lengths
parser.add_argument('--delta', type=float, default=9.0)
# vcdr groundtruth file
parser.add_argument('--gt', dest='vcdr_gt_filename', type=str, default="SEED_vcdr.csv")
# predicted vcdr save file
parser.add_argument('--savepred', dest='vcdr_pred_save_filename', type=str, default="vcdr_pred.csv")
args = parser.parse_args()

print(f"Use delta={args.delta}")
subj2LR2vcdr = {}

for mask_filename in listdir(args.maskdir):
    subj, LR = mask_filename[:-4].split("-")
    mask_path = join(args.maskdir, mask_filename)
    mask_obj  = Image.open(mask_path, 'r')
    mask = np.array(mask_obj)
    mask_nhot = fundus_map_mask(mask)
    vcdr = calc_vcdr(torch.tensor(mask_nhot), delta=args.delta)
    if vcdr == -1:
        continue
    if subj not in subj2LR2vcdr:
        subj2LR2vcdr[subj] = {}
    subj2LR2vcdr[subj][LR] = vcdr
    # print("{},{:.3f}".format(mask_filename, vcdr.item()))

subjects = sorted(subj2LR2vcdr.keys())
VCDR = open(args.vcdr_pred_save_filename, "w")
VCDR.write("sno,vertcdrr,vertcdrl\n")

left_count, right_count = 0, 0
for subj in subjects:
    if 'L' in subj2LR2vcdr[subj]:
        vcdr_l = "{:.5f}".format(subj2LR2vcdr[subj]['L'])
        left_count += 1
    else:
        vcdr_l = ""

    if 'R' in subj2LR2vcdr[subj]:
        vcdr_r = "{:.5f}".format(subj2LR2vcdr[subj]['R'])
        right_count += 1
    else:
        vcdr_r = ""    

    VCDR.write(f"{subj},{vcdr_r},{vcdr_l}\n")

print(f"{left_count} left eyes, {right_count} right eyes saved to ‘{args.vcdr_pred_save_filename}’")
calc_vcdr_acc(args.vcdr_gt_filename, args.vcdr_pred_save_filename)
