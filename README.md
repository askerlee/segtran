# README

# Medical Image Segmentation using Squeeze-and-Expansion Transformers

### & [Few-Shot Domain Adaptation with Polymorphic Transformers](README_polyformer.md)

### BUG FIXES in the 3D pipeline

Sorry in the initial release, there were a few bugs preventing training on 3D images. These were caused by obsolete code pieces. Now they are fixed. Please `git pull origin mater` to update the code.

### Introduction

This repository contains the code of the IJCAI'2021 paper [**Medical Image Segmentation using Squeeze-and-Expansion Transformers**](https://arxiv.org/abs/2105.09511) & the MICCAI'2021 paper **Few-Shot Domain Adaptation with Polymorphic Transformers**.

### Installation

This repository is based on PyTorch>=1.7.

`pip install -r requirements.txt`

If you'd like to evaluate setr, you need to install mmcv according to https://github.com/fudan-zvg/SETR/.

### Usage Example

The examples for **Polymorphic Transformers (Polyformer)** can be found [here](README_polyformer.md).

### A 2D segmentation task:

`python3 train2d.py --task refuge --split all --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --maxiter 10000`

`python3 test2d.py  --task refuge --split all --ds valid2 --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --cpdir ../model/segtran-refuge-train,valid,test,drishiti,rim-05101448 --iters 7000 --outorigsize`

### A 3D segmentation task:

`python3 train3d.py --task brats --split all --bs 2 --maxiter 10000 --randscale 0.1 --net segtran --attractors 1024 --translayers 1`

`python3 test3d.py --task brats --split all --bs 5 --ds 2019valid --net segtran --attractors 1024 --translayers 1 --cpdir ../model/segtran-brats-2019train-01170142 --iters 8000`

### Acknowledgement

The "Receptivefield" folder is from https://github.com/fornaxai/receptivefield/, with minor edits and bug fixes.

The "MNet\_DeepCDR" folder is from https://github.com/HzFu/MNet_DeepCDR, with minor customizations.

The "Efficientnet" folder is from https://github.com/lukemelas/EfficientNet-PyTorch, with minor customizations.

The "networks/setr" folder is a slimmed-down version of https://github.com/fudan-zvg/SETR/, with a few custom config files.

There are a few baseline models under networks/ which were originally implemented in various github repos. Here I won't acknowlege them individually.

Some code under "dataloaders" (esp. 3D image preprocessing) was borrowed from https://github.com/yulequan/UA-MT.

### Citations

If you find our code useful, please kindly consider to cite one of our papers as:

```
@InProceedings{segtran,
author="Li, Shaohua and Sui, Xiuchao and Luo, Xiangde and Xu, Xinxing and Liu Yong and Goh, Rick Siow Mong",
title="Medical Image Segmentation using Squeeze-and-Expansion Transformers",
booktitle="The 30th International Joint Conference on Artificial Intelligence (IJCAI)",
year="2021"}

@InProceedings{polyformer,
author="Li, Shaohua and Sui, Xiuchao and Fu, Jie and Fu, Huazhu and Luo, Xiangde and Feng, Yangqin and Xu, Xinxing and Liu Yong and Ting, Daniel and Goh, Rick Siow Mong",
title="Few-Shot Domain Adaptation with Polymorphic Transformers",
booktitle="The 24th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)",
year="2021"}
```
