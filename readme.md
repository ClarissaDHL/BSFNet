# BSFNet: Bilateral Semantic Fusion Siamese Network for Change Detection from Multi-Temporal Optical Remote Sensing Imagery
Change detection is an essential task in optical remote sensing, and it can be used to extract the valid information from sequential multi-temporal images. However, since the character of long-term revisiting and very high resolution (VHR) development, the great differences of illumination, season and interior textures between bi-temporal images bring considerable 
challenges for pixel-wise change detection. In this letter, focus on accurate pixel-wise change detection, a bilateral semantic fusion Siamese network (BSFNet) is proposed. Firstly, to better map bi-temporal images into semantic feature domain for comparison, a novel bilateral semantic fusion Siamese network is designed to effectively integrate shallow and deep semantic features, which can provide pixel-wise change detection results with complete regions and clear boundary locations. Then, in order to facilitatethe reasonable convergence of the proposed BSFNet, a scaleinvariant sample balance (SISB) loss is designed for metric learning to avoid the problems of sample imbalance and scale variance. Finally, extensive experiments are carried out on twopublished CDD and LEVIR change detection datasets, and results indicate that the proposed BSFNet can provide superior performance than the other state-of-the-art methods.
# Requirement
Pytorch 1.2.0<br>
torchvision <br>
python 3.7.4 <br>
# Datasets
The experiments of our work are carried out on published CDD and LEVIR_CD datasets. <br>
## CDD
papar: [CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS](https://www.researchgate.net/publication/325470033_CHANGE_DETECTION_IN_REMOTE_SENSING_IMAGES_USING_CONDITIONAL_ADVERSARIAL_NETWORKS) <br>
## LEVIR_CD
paper: [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://justchenhao.github.io/LEVIR/) <br>
## Directory Structure
$T0_image_path/*.jpg(.png)
$T1_image_path/*.jpg(.png)
$ground_truth_path/*.jpg(.png)
# Pretrained Model
You can download the best checkpoint from [baidudisk](https://pan.baidu.com/s/1M-hmvYyUPEkk5fcWTucdDw)password:RSCD
# model training
cd_code/bsfnet_cd.py
# model testing
cd_code/test_code.py
# Citation
