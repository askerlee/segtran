# Medical Image Segmentation using Squeeze-and-Expansion Transformers
### Introduction

This repository contains the code of the IJCAI'2021 paper 'Medical Image Segmentation using Squeeze-and-Expansion Transformers'. 

### Installation
This repository is based on PyTorch 1.7. 

To evaluate setr, you need to install mmcv according to https://github.com/fudan-zvg/SETR/.

### Usage Example

#### A 2D task:
`python3.7 train2d.py --task refuge --split all --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --maxiter 10000`

`python3.7 test2d.py  --task refuge --split all --ds valid2 --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --cpdir ../model/segtran-refuge-train,valid,test,drishiti,rim-05101448 --iters 7000 --outorigsize`

#### A 3D task:
`python3.7 train3d.py --task brats --split all --bs 2 --maxiter 10000 --randscale 0.1 --net segtran --attractors 1024 --translayers 1`

`python3.7 test3d.py --task brats --split all --bs 5 --ds 2019valid --net segtran --attractors 1024 --translayers 1 --cpdir ../model/segtran-brats-2019train-01170142 --iters 8000`

### Acknowledgement
The "receptivefield" folder is from https://github.com/fornaxai/receptivefield/, with minor edits and bug fixes.

The "MNet\_DeepCDR" folder is from https://github.com/HzFu/MNet_DeepCDR, with minor customizations.

The "efficientnet" folder is from https://github.com/lukemelas/EfficientNet-PyTorch, with minor customizations.

The "networks/setr" folder is a slimmed-down version of https://github.com/fudan-zvg/SETR/, with a few custom config files.

There are a few baseline models under networks/ which were originally implemented in various github repos. Here I won't acknowlege them individually.

Some code under "dataloaders/" (esp. 3D image preprocessing) was borrowed from https://github.com/yulequan/UA-MT.

### Citation
If you find our code useful, please kindly consider to cite our paper as:
```bibtex
@InProceedings{segtran,
author="Li, Shaohua
and Sui, Xiuchao
and Luo, Xiangde
and Xu, Xinxing
and Liu Yong
and Goh, Rick Siow Mong",
title="Medical Image Segmentation using Squeeze-and-Expansion Transformers",
booktitle="The 30th International Joint Conference on Artificial Intelligence (IJCAI)",
year="2021",
}
```
