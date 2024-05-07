# DuAT
Feilong Tang, Qiming Huang, Jinfeng Wang, Xianxu Hou, Jionglong Su, and Jingxin Liu

This repo is the official implementation of ["DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation"](https://arxiv.org/abs/2212.11677). 



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-2018-data)](https://paperswithcode.com/sota/medical-image-segmentation-on-2018-data?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/lesion-segmentation-on-isic-2018)](https://paperswithcode.com/sota/lesion-segmentation-on-isic-2018?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=duat-dual-aggregation-transformer-network-for)


## 1. Introduction
**DuAT** is initially described in [PRCV](https://arxiv.org/pdf/2212.11677.pdf).

Transformer-based models have been widely demon- strated to be successful in computer vision tasks by mod- elling long-range dependencies and capturing global rep- resentations. However, they are often dominated by fea- tures of large patterns leading to the loss of local details (e.g., boundaries and small objects), which are critical in medical image segmentation. To alleviate this problem, we propose a Dual-Aggregation Transformer Network called DuAT, which is characterized by two innovative designs, namely, the Global-to-Local Spatial Aggregation (GLSA) and Selective Boundary Aggregation (SBA) modules. The GLSA has the ability to aggregate and represent both global and local spatial features, which are beneficial for locat- ing large and small objects, respectively. The SBA mod- ule is used to aggregate the boundary characteristic from low-level features and semantic information from high-level features for better preserving boundary details and locat- ing the re-calibration objects. Extensive experiments in six benchmark datasets demonstrate that our proposed model outperforms state-of-the-art methods in the segmentation of skin lesion images, and polyps in colonoscopy images. In addition, our approach is more robust than existing meth- ods in various challenging situations such as small object segmentation and ambiguous object boundaries.


## 2. Framework Overview
![](https://github.com/Barrett-python/DuAT/blob/main/Fig/fig1.png)

## 3. Results
### 3.1 Image-level Polyp Segmentation
![](https://github.com/Barrett-python/DuAT/blob/main/Fig/fig2.png)
The polyp Segmentation prediction results in [here](https://drive.google.com/drive/folders/14IDwewAb12HWlxgOFtFB46aMJyqPaKpz?usp=sharing). 

## 4. Usage:
### 4.1 Recommended environment:
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```
### 4.2 Data preparation:
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1OBVivLJAs9ZpnB5I2s3lNg) [code:dr1h].


### 4.3 Pretrained model:
You should download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1Vez7iT2v_g7VYsDxRGE8HA) [code:w4vk], and then put it in the './pretrained_pth' folder for initialization. 

### 4.4 Training:
Clone the repository:
```
git clone https://github.com/Barrett-python/DuAT.git
cd DuAT
bash train.sh
```

### 4.5 Testing:
```
cd DuAT
bash test.sh
```


### 4.6 Evaluating your trained model:

Matlab: Please refer to the work of MICCAI2020 ([link](https://github.com/DengPingFan/PraNet)).

Python: Please refer to the work of ACMMM2021 ([link](https://github.com/plemeri/UACANet)).

Please note that we use the Matlab version to evaluate in our paper.


### 4.7 Well trained model:
You could download the trained model from [Google Drive](https://drive.google.com/drive/folders/14IDwewAb12HWlxgOFtFB46aMJyqPaKpz) and put the model in directory './model_pth'.


Citation If you find this code or idea useful, please cite our work:
## Citation:
```
@inproceedings{tang2023duat,
  title={DuAT: Dual-aggregation transformer network for medical image segmentation},
  author={Tang, Feilong and Xu, Zhongxing and Huang, Qiming and Wang, Jinfeng and Hou, Xianxu and Su, Jionglong and Liu, Jingxin},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={343--356},
  year={2023},
  organization={Springer}
}
```


## 6. Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [SSformer](https://github.com/Qiming-Huang/ssformer), which have provided the basis for our framework.

<!-- ## 7. FAQ:
If you want to improve the usability or any piece of advice, please feel free to contact me directly. -->

