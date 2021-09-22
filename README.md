# Medical Image Segmentation using Squeeze-and-Expansion Transformers

## & [Few-Shot Domain Adaptation with Polymorphic Transformers](README_polyformer.md)

### 09/19/2021  Segtran checkpoints trained on REFUGE 2020 (2D fundus images) and BraTS 2019 (3D Brain MRI):

[https://pan.baidu.com/s/1nbBrPJKK1NtOe848cApS8w](https://pan.baidu.com/s/1nbBrPJKK1NtOe848cApS8w)

code: h1vi

On BraTS 2019 validation: ET 0.729 / WT 0.896 / TC 0.832, Avg. 0.819

On REFUGE 2020 validation: Cup 0.870 / Disc 0.959, Avg. 0.915

They are newly trained, and the performance is slightly different from reported in paper (BraTS is higher and REFUGE is lower).

REFUGE training command line (with --noqkbias the trained model performed slightly better):

`./train2d.sh --task fundus --split all --translayers 3 --layercompress 1,1,2,2 --net segtran --bb eff-b4 --maxiter 10000 --bs 6 --noqkbias`

(The checkpoint above is iter_5000.pth.)

BraTS training command line (with --noqkbias the trained model performed slightly worse):

`./train3d.sh --split all --maxiter 10000 --task brats --translayers 1 --bs 4 --randscale 0.1 --attractors 1024`

(The checkpoint above is iter_8000.pth.)

### 06/10/2021  BUG FIXES in the 3D pipeline

Sorry in the initial release, there were a few bugs preventing training on 3D images. These were caused by obsolete code pieces. Now they are fixed. Please `git pull origin mater` to update the code.

### Introduction

This repository contains the code of the IJCAI'2021 paper 

- **[Medical Image Segmentation using Squeeze-and-Expansion Transformers](https://arxiv.org/abs/2105.09511)**

    Medical image segmentation is important for computer-aided diagnosis. Good segmentation demands the model to see the big picture and fine details simultaneously, i.e., to learn image features that incorporate large context while keep high spatial resolutions. To approach this goal, the most widely used methods -- U-Net and variants, extract and fuse multi-scale features. However, the fused features still have small "effective receptive fields" with a focus on local image cues, limiting their performance. In this work, we propose Segtran, an alternative segmentation framework based on transformers, which have unlimited "effective receptive fields" even at high feature resolutions. The core of Segtran is a novel Squeeze-and-Expansion transformer: a squeezed attention block regularizes the self attention of transformers, and an expansion block learns diversified representations. Additionally, we propose a new positional encoding scheme for transformers, imposing a continuity inductive bias for images. Experiments were performed on 2D and 3D medical image segmentation tasks: optic disc/cup segmentation in fundus images (REFUGE'20 challenge), polyp segmentation in colonoscopy images, and brain tumor segmentation in MRI scans (BraTS'19 challenge). Compared with representative existing methods, Segtran consistently achieved the highest segmentation accuracy, and exhibited good cross-domain generalization capabilities. The source code of Segtran is released at [https://github.com/askerlee/segtran](https://github.com/askerlee/segtran).

and the MICCAI'2021 paper 

- **[Few-Shot Domain Adaptation with Polymorphic Transformers](https://arxiv.org/abs/2107.04805)**.

    Deep neural networks (DNNs) trained on one set of medical images often experience severe performance drop on unseen test images, due to various domain discrepancy between the training images (source domain) and the test images (target domain), which raises a domain adaptation issue. In clinical settings, it is difficult to collect enough annotated target domain data in a short period. Few-shot domain adaptation, i.e., adapting a trained model with a handful of annotations, is highly practical and useful in this case. In this paper, we propose a Polymorphic Transformer (Polyformer), which can be incorporated into any DNN backbones for few-shot domain adaptation. Specifically, after the polyformer layer is inserted into a model trained on the source domain, it extracts a set of prototype embeddings, which can be viewed as a "basis" of the source-domain features. On the target domain, the polyformer layer adapts by only updating a projection layer which controls the interactions between image features and the prototype embeddings. All other model weights (except BatchNorm parameters) are frozen during adaptation. Thus, the chance of overfitting the annotations is greatly reduced, and the model can perform robustly on the target domain after being trained on a few annotated images. We demonstrate the effectiveness of Polyformer on two medical segmentation tasks (i.e., optic disc/cup segmentation, and polyp segmentation). The source code of Polyformer is released at [https://github.com/askerlee/segtran](https://github.com/askerlee/segtran).

### Datasets

The `refuge` datasets, i.e., the `train`, `valid`, `test` splits of refuge, can be downloaded from [https://refuge.grand-challenge.org/Download/](https://refuge.grand-challenge.org/Download/) (after registration). The `RIM-One` and `Drishti-GS` (not used for DA) datasets can be downloaded from [http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/](http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/) and [https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php), respectively.

The `polyp` datasets, i.e., `CVC-ClinicDB` (a.k.a. `CVC612`), `Kvasir`, `CVC-300`, `CVC-ColonDB`, `ETIS-LaribPolypDB` can be downloaded from [https://github.com/DengPingFan/PraNet](https://github.com/DengPingFan/PraNet) (search for "testing data"). 

### Installation

This repository is based on PyTorch>=1.7.

```bash
git clone [https://github.com/askerlee/segtran](https://github.com/askerlee/segtran)
cd segtran
download data...
pip install -r requirements.txt
cd code
python3 train2d.py/test2d.py/train3d.py/test3d.py ...
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

`--net`: which type of segmentation model to use. Currently more than 10 types of 2D segmentation models can be chosen from. `unet`: U-Net with pretrained CNN encoder. `unet-scratch`: vanilla U-Net. `nestedunet`: Nested U-Net. `unet3plus`: U-Net 3+. `pranet`: PraNet. `attunet`: Attention U-Net. r2attunet: a combination of attention U-Net and Recurrent Residual U-Net. `dunet`: deformable U-Net. `setr`: SEgmentation TRansformer. `transunet`: U-Net with a transformer encoder. `deeplab`: DeepLabv3+. `nnunet`: nnU-Net (only the model, not the whole pipeline). `segtran`: Squeeze-and-Expansion transformer for segmentation.

`--bb`: the type of CNN backbone/encoder. Commonly used 2D backbones are `eff-b1, ..., eff-b4` (EfficientNet-B1~B4) and `resnet101`. 

`--translayers`: the number of transformer layers (only used with `--net segtran`).

`--layercompress`: the ratios of transformer channel compression done in different layers.  Channel compression means the number of output channels could be fewer than the input channels, so as to reduce the number of parameters and reduce the chance of overfitting. Example format: `1,1,2,2`. This constraint should be satisfied: `len(layercompress) == translayers + 1`. The first number is the compression ratio between the CNN backbone and the transformer input. If `layercompress[0] > 1`, then a bridging conv layer will be used to reduce the output feature dimension of the CNN backbone.  If `layercompress[i] > 1, i >=1`, then the transformer will output lower-dimensional features. 

`--maxiter`: the maximum number of iterations. For refuge, maxiter is usually `10000` (the optimal checkpoint is usually around `7000` iterations). For polyp, maxiter is usually `14000` (the optimal checkpoint is usually around `13000` iterations).

`--iters`: which iteration(s) of checkpoints to load and test. `7000,8000`: load and test iterations `7000` and `8000`. `5000-10000,1000`: load iterations of `range(5000, 10000+1000, 1000)`, i.e., `5000, 6000, 7000, 8000, 9000, 10000`.

### A 3D segmentation task:

`python3 train3d.py --task brats --split all --bs 2 --maxiter 10000 --randscale 0.1 --net segtran --attractors 1024 --translayers 1`

`python3 test3d.py --task brats --split all --bs 5 --ds 2019valid --net segtran --attractors 1024 --translayers 1 --cpdir ../model/segtran-brats-2019train-01170142 --iters 8000`

*Arguments:*

`--net`: which type of model to use. Currently three 3D segmentation models can be chosen from. `unet`: 3D U-Net. `vnet`: V-Net. `segtran`: Squeeze-and-Expansion transformer for segmentation.

`--bb`: the type of CNN backbone for `segtran`. A commonly used 3D backbone is `i3d` (default).

`--attractors`: the number of attractors in the Squeezed Attention Block. 

To save GPU RAM, 3D tasks usually only use one transformer layer, i.e., `--translayers 1`.

### Data Preparation

For 2D fundus images, please use `MNet_DeepCDR/Step_1_Disc_Crop.py` to crop out the optic disc area from each image. You need to manually edit the source image paths in this file.

For 3D MRI images, please use `dataloaders/brats_processing.py` to convert a folder of MRI images to .h5:

```bash
python dataloaders/brats_processing.py h5 ../data/BraTS2019_Training
python dataloaders/brats_processing.py h5 ../data/BraTS2019_Validation
```

For BraTS* datasets, please use a folder name containing "**validation**" (case insensitive) to store the validation data, so that `brats_processing.py` knows those are validation data and would not try to separate a channel out as the segmentation mask.

### Acknowledgement

The "receptivefield" folder is from https://github.com/fornaxai/receptivefield/, with minor edits and bug fixes.

The "MNet_DeepCDR" folder is from https://github.com/HzFu/MNet_DeepCDR, with minor customizations.

The "efficientnet" folder is from https://github.com/lukemelas/EfficientNet-PyTorch, with minor customizations.

The "networks/setr" folder is a slimmed-down version of https://github.com/fudan-zvg/SETR/, with a few custom config files.

There are a few baseline models under networks/ which were originally implemented in various github repos. Here I won’t acknowlege them individually.

Some code under "dataloaders/" (esp. 3D image preprocessing) was borrowed from https://github.com/yulequan/UA-MT.

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