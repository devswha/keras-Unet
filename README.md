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

<p align="center">
<img src="https://github.com/devswha/keras-Unet/blob/master/images/sliding_window.png" width="30%" height="30%"> <br/>Sliding window</td>
</p>

<p align="center">
<img src="https://github.com/devswha/keras-Unet/blob/master/images/patch.png" width="50%" height="50%"> <br/>Patch</td>
</p>


- Patch method has low overlap ratio so that the speed of detection can be improvement.
- However, as the wide size of patch detect image at once, the performance of context is good but the performance of localization is lower. 
- In this paper, the U-net architecture and overlap-tile methods were proposed to solve this localization problem.


<img src="https://github.com/devswha/keras-Unet/blob/master/images/overlap_tile.png" width="100%" height="100%"> <br/>Overlap-tile</td>

Simple. Because the EM image is large, sometimes the model of detection input is larger than the patch size (yellow). If so, mirror and fill in the patch area with the empty part.
