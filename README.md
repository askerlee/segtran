# Medical Image Segmentation using Squeeze-and-Expansion Transformers

## & [Few-Shot Domain Adaptation with Polymorphic Transformers](README_polyformer.md)

### BUG FIXES in the 3D pipeline

Sorry in the initial release, there were a few bugs preventing training on 3D images. These were caused by obsolete code pieces. Now they are fixed. Please `git pull origin mater` to update the code.

### Introduction

This repository contains the code of the IJCAI’2021 paper **Medical Image Segmentation using Squeeze-and-Expansion Transformers** & the MICCAI'2021 paper **Few-Shot Domain Adaptation with Polymorphic Transformers**.

### Datasets

The refuge datasets (the "train", "valid", "test" splits of refuge) can be downloaded from [https://refuge.grand-challenge.org/Download/](https://refuge.grand-challenge.org/Download/) (after registration). The RIM-One and Drishti-GS (not used for DA) datasets can be downloaded from [http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/](http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/) and [https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php), respectively.

The Polyp datasets (CVC612, Kvasir and CVC-300) can be downloaded from [https://github.com/DengPingFan/PraNet](https://github.com/DengPingFan/PraNet) (search for "testing data"). 

### Installation

This repository is based on PyTorch>=1.7.

```bash
git clone [https://github.com/askerlee/segtran](https://github.com/askerlee/segtran)
cd segtran
download data...
pip install -r requirements.txt
cd code
python3 train2d.py/test2d.py/train3d.py/test3d.py...
```

If you'd like to evaluate setr, you need to install mmcv according to [https://github.com/fudan-zvg/SETR/](https://github.com/fudan-zvg/SETR/).

### Usage Example

The examples for **Polymorphic Transformers (Polyformer)** can be found [here](README_polyformer.md).

### A 2D segmentation task:

`python3 train2d.py --task refuge --split all --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --maxiter 10000`

`python3 test2d.py  --task refuge --split all --ds valid2 --net segtran --bb resnet101 --translayers 3 --layercompress 1,1,2,2 --cpdir ../model/segtran-refuge-train,valid,test,drishiti,rim-05101448 --iters 7000 --outorigsize`

*Arguments:*

`--task`: the segmentation task to work on. Supported tasks are hard-coded in train2d.py/test2d.py/train3d.py/test3d.py. Currently three 2D tasks are built-in: `refuge`, `polyp` and `oct`; two 3D tasks are built-in: `brats` and `atria`.

`--ds`: dataset(s) to use for training/test. If not specified for training, the default training datasets for the current task will be used. For refuge, the default are `train, valid, test, drishiti, rim`. For polyp, the default are `CVC-ClinicDB-train, Kvasir-train`.

`--split`: which part(s) of the dataset(s) to use. `all`: use the whole dataset(s). `train`: use the "train" split (usually random 85% of the whole dataset). `test`: use the "test" split (usually the remaining 15% of the whole dataset). The split is done in `dataloaders/{datasets2d.py, datasets3d.py}`. 

`--net`: which type of model to use. Currently more than 10 types of 2D segmentation models can be chosen from. `unet`: U-Net with pretrained CNN encoder. `unet-scratch`: vanilla U-Net. `nestedunet`: Nested U-Net. `unet3plus`: U-Net 3+. `pranet`: PraNet. `attunet`: Attention U-Net. r2attunet: a combination of attention U-Net and Recurrent Residual U-Net. `dunet`: deformable U-Net. `setr`: SEgmentation TRansformer. `transunet`: U-Net with a transformer encoder. `deeplab`: DeepLabv3+. `nnunet`: nnU-Net (only the model, not the whole pipeline). `segtran`: Squeeze-and-Expansion transformer for segmentation.

`--bb`: the type of the backbone/encoder. Commonly used 2D backbones are `eff-b1, ..., eff-b4` (EfficientNet-B1~B4) and `resnet101`. Commonly used 3D backbone is `i3d`.

`--translayers`: number of transformer layers (only used with `--net segtran`).

`--layercompress`: the ratios of transformer channel compression done in different layers.  Channel compression means the number of output channels could be fewer than the input channels, so as to reduce the number of parameters and reduce the chance of overfitting. Example format: `1,1,2,2`. This constraint should be satisfied: `len(layercompress) == translayers + 1`. The first number is the compression ratio between the CNN backbone and the transformer input. If `layercompress[0] > 1`, then a bridging conv layer will be used to reduce the output feature dimension of the CNN backbone.  If `layercompress[i] > 1, i >=1`, then the transformer will output lower-dimensional features. 

`--maxiter`: the maximum number of iterations. For refuge, maxiter is usually 10000 (the optimal checkpoint is around 7000 iterations). For polyp, maxiter is usually 14000 (the optimal checkpoint is around 13000 iterations).

`--iters`: which iteration(s) of checkpoints to load and test. `7000,8000`: load and test iterations `7000` and `8000`. `5000-10000,1000`: load iterations of `range(5000, 10000+1000, 1000)`, i.e., `5000, 6000, 7000, 8000, 9000, 10000`.

### A 3D segmentation task:

`python3 train3d.py --task brats --split all --bs 2 --maxiter 10000 --randscale 0.1 --net segtran --attractors 1024 --translayers 1`

`python3 test3d.py --task brats --split all --bs 5 --ds 2019valid --net segtran --attractors 1024 --translayers 1 --cpdir ../model/segtran-brats-2019train-01170142 --iters 8000`

*Arguments:*

`--attractors`: the number of attractors in the Squeezed Attention Block. 

To save GPU RAM, 3D tasks usually only use one transformer layer, i.e., `--translayers 1`.

### Acknowledgement

The “receptivefield” folder is from https://github.com/fornaxai/receptivefield/, with minor edits and bug fixes.

The “MNet_DeepCDR” folder is from https://github.com/HzFu/MNet_DeepCDR, with minor customizations.

The “efficientnet” folder is from https://github.com/lukemelas/EfficientNet-PyTorch, with minor customizations.

The “networks/setr” folder is a slimmed-down version of https://github.com/fudan-zvg/SETR/, with a few custom config files.

There are a few baseline models under networks/ which were originally implemented in various github repos. Here I won’t acknowlege them individually.

Some code under “dataloaders/” (esp. 3D image preprocessing) was borrowed from https://github.com/yulequan/UA-MT.

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

[Few-Shot Domain-Adaptation with **Polymorphic Transformers**](README_polyformer.md)