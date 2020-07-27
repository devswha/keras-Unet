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


- Electron Microscopic (EM) 이미지 segmentation 으로 membrane 과 cell 의 black and white segmentation
- EM(전자현미경) 이미지라서 사이즈가 크고 data 가 몇 없어서 (30장) data augmentation 이 필요
- black (1) white (0) 으로 masking