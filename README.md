# Slice-Loc
![](figures/Fig1.jpg)
![](figures/Fig9.jpg)

## ✅ To-Do

- [x] Initial repo structure
- [ ] DReSS-D Dataset
- [ ] Training scripts
- [ ] Testing scripts

## 1. Cross-View Localization via Redundant Sliced Observations and A-Contrario Validation
Cross-view localization (CVL) matches ground-level images with aerial references to determine the geo-position of a camera, enabling smart vehicles to self-localize offline in GNSS-denied environments. However, most CVL methods output only a single observation, the camera pose, and lack the redundant observations required by surveying principles, making it challenging to assess localization reliability through the mutual validation of observational data. To tackle this, we introduce Slice-Loc, a two-stage method featuring an a-contrario reliability validation for CVL. Instead of using the query image as a single input, Slice-Loc divides it into sub-images and estimates the 3-DoF pose for each slice, creating redundant and independent observations. Then, a geometric rigidity formula is proposed to filter out the erroneous 3-DoF poses, and the inliers are merged to generate the final camera pose. Furthermore, we propose a model that quantifies the meaningfulness of localization by estimating the number of false alarms (NFA), according to the distribution of the locations of the sliced images.

## 2. DReSS-D Dataset
Building on DReSS, the DReSS-D dataset is provided,which includes ground and aerial images from six cities across six continents: Sydney, Chicago, Johannesburg, Tokyo, Rio, and London. DReSS-D provides a depth map for each panoramic image.

### 2.1 DReSS Dataset
DReSS: [Baidu Netdisk](https://pan.baidu.com/s/1m3VLsyX3mIl1DmK_X6v4Lw?pwd=MAgs), [Huggingface part-1](https://huggingface.co/datasets/SummerpanKing/DReSS-part1), [Huggingface-part2](https://huggingface.co/datasets/Mabel0403/DReSS-part2).

## Acknowledgments

This code is based on the amazing work of: [CCVPE](https://github.com/tudelft-iv/CCVPE), [HC-Net](https://github.com/xlwangDev/HC-Net) and [AuxGeo](https://github.com/SummerpanKing/DReSS). We appreciate the previous open-source works.

