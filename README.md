# keras-Unet

The implementation of biomedical image segmentation with the use of U-Net model. The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Abstract

The author of paper propose a simple and effective end-to-end image segmentation network architecture for medical images.
The proposed network, called U-net, has main three factors for well-training.
- U-shaped network structure with two configurations: Contracting and Expanding path
- Training more faster than sliding-windows: Patch units and Overlap-tile
- Data augmentation: Elastic deformation and Weight cross entropy

## Dataset

The dataset we used is Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC), which is dowloaded from [ISBI Challenge: Segmentation of of neural structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/home)


<p align="center">
    <img src="https://github.com/devswha/keras-Unet/blob/master/images/ISBI.gif">


- Black and white segmentation of membrane and cell with EM(Electron Microscopic) image.
- The data set is a large size of image and few so the data augmentation is needed.
- The data set contains 30 images of size 512x512 for the train, train-labels and test.
- There is no images for test-labels for the ISBI competition. 
- If you want to get the evaluation metrics of competition, you should split part of the train data set for testing.

## Overlap-tile

Sliding window
<img src="https://github.com/devswha/keras-Unet/blob/master/images/sliding_window.png" width="30%" height="30%">

Patch
<img src="https://github.com/devswha/keras-Unet/blob/master/images/patch.png" width="30%" height="30%">



- Overlap 비율이 적기 때문에 속도 향상이 가능하다. 
- 하지만 patch size 가 넓은 범위의 이미지를 한번에 인식하다보니 context(이미지 픽셀 간 관계) 성능은 좋지만 localization 성능이 낮아진다. 
- 따라서 해당 논문에서는 이 localization 문제를 해결하기 위해 Unet architecture 과 overlap-tile 방식을 제안하였다.


Overlap-tile
<img src="./overlap_tile.png" width="100%" height="100%">

- 전체 이미지는 크다보니 각 patch 별로 이미지를 인식하는데, patch 크기(노란색) 보다 Input이 클 경우가 있다.
- 그럴 경우, 비어있는 부분을 patch 영역을 미러링하여 채워넣는다
