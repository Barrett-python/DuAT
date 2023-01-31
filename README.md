# DuAT
Feilong Tang, Qiming Huang, Jinfeng Wang, Xianxu Hou, Jionglong Su, and Jingxin Liu

This repo is the official implementation of ["DuAT: DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation"](https://arxiv.org/abs/2212.11677). 

## 1. Introduction
**DuAT** is initially described in [arxiv](https://arxiv.org/pdf/2212.11677.pdf).

Transformer-based models have been widely demon- strated to be successful in computer vision tasks by mod- elling long-range dependencies and capturing global rep- resentations. However, they are often dominated by fea- tures of large patterns leading to the loss of local details (e.g., boundaries and small objects), which are critical in medical image segmentation. To alleviate this problem, we propose a Dual-Aggregation Transformer Network called DuAT, which is characterized by two innovative designs, namely, the Global-to-Local Spatial Aggregation (GLSA) and Selective Boundary Aggregation (SBA) modules. The GLSA has the ability to aggregate and represent both global and local spatial features, which are beneficial for locat- ing large and small objects, respectively. The SBA mod- ule is used to aggregate the boundary characteristic from low-level features and semantic information from high-level features for better preserving boundary details and locat- ing the re-calibration objects. Extensive experiments in six benchmark datasets demonstrate that our proposed model outperforms state-of-the-art methods in the segmentation of skin lesion images, and polyps in colonoscopy images. In addition, our approach is more robust than existing meth- ods in various challenging situations such as small object segmentation and ambiguous object boundaries.


## 2. Framework Overview
![](https://github.com/Barrett-python/DuAT/Fig/fig1.png)



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-2018-data)](https://paperswithcode.com/sota/medical-image-segmentation-on-2018-data?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/lesion-segmentation-on-isic-2018)](https://paperswithcode.com/sota/lesion-segmentation-on-isic-2018?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=duat-dual-aggregation-transformer-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/duat-dual-aggregation-transformer-network-for/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=duat-dual-aggregation-transformer-network-for)


The network part has been uploaded!

Citation If you find this code or idea useful, please cite our work:


## Citation:
```
@article{tang2022duat,
  title={DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation},
  author={Tang, Feilong and Huang, Qiming and Wang, Jinfeng and Hou, Xianxu and Su, Jionglong and Liu, Jingxin},
  journal={arXiv preprint arXiv:2212.11677},
  year={2022}
}
```



