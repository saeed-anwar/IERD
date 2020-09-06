# Identity Enhanced Residual Image Denoising
This repository is for Identity Enhanced Residual Image Denoising (IERD) introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/),  [Cong Phuoc Huynh](https://www.linkedin.com/in/cong-phuoc-huynh-61891b15/), [Fatih Porikli](http://porikli.com/), "[Identity Enhanced Residual Image Denoising](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Anwar_Identity_Enhanced_Residual_Image_Denoising_CVPRW_2020_paper.pdf)", IEEE Computer Vision and Pattern Recognition Workshop, CVPRw, 2020



## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Requirements](#requirements)
4. [Test](#test)
5. [Results](#results)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Introduction
We propose to learn a fully-convolutional network model that consists of a Chain of Identity Mapping Modules and residual on the residual architecture for image denoising.
Our network structure possesses three distinctive features that are important for the noise removal task. Firstly, each unit employs identity mappings as the skip connections and receives pre-activated input to preserve the gradient magnitude propagated in both the forward and backward directions. Secondly, by utilizing dilated kernels for the convolution layers in the residual branch, each neuron in the last convolution layer of each module can observe the full receptive field of the first layer. Lastly, we employ the residual on the residual architecture to ease the propagation of the high-level information. Contrary to current state-of-the-art real denoising networks, we also present a straightforward and single-stage network for real image denoising.

The proposed network produces remarkably higher numerical accuracy and better visual image quality than the classical state-of-the-art and CNN algorithms when being evaluated on the three conventional benchmark and three real-world datasets

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Main.png">
</p>

Denoising results: In the first row, an image is corrupted by the Gaussian noise with σ = 50 from the BSD68 dataset. In the second row, a sample image from the RNI15 real noisy dataset. Our results have the best PSNR score for synthetic images, and unlike other methods, it does not have over-smoothing or over-contrasting artifacts. 

## Network
The proposed network architecture, which consists of multiple modules with similar structures. Each module is composed of a series of pre-activation-convolution layer pairs. The multiplier block negates the input block features to be summed at the end of the mapping module

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Network.png">
</p>


## Requirements
The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The real denoising model can be downloaded from [Google Drive](https://drive.google.com/open?id=1QxO6KFOVxaYYiwxliwngxhw_xCtInSHd) or [here](https://icedrive.net/0/e3Cb4ifYSl). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
    ```


## Results
**All the results for IERD can be downloaded from GoogleDrive from [SSID](), [RNI15]() and [DnD](). The size of the results is 65MB** 

### Quantitative Results
<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Network.png">
</p>
The performance of state-of-the-art algorithms on widely used publicly available DnD dataset in terms of PSNR (in dB) and SSIM. The best results are highlighted in bold.

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Network.png">
</p>
The quantitative results (in PSNR (dB)) for the SSID and Nam datasets.. The best results are presented in bold.

For more information, please refer to our [papar](https://arxiv.org/abs/1904.07396)

### Visual Results
<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Network.png">
</p>

A real noisy example from DND dataset for comparison of our method against the state-of-the-art algorithms.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{anwar2020IERD,
  title={Identity Enhanced Residual Image Denoising},
  author={Anwar, Saeed and Phuoc Huynh, Cong and Porikli, Fatih},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={520--521},
  year={2020}
}

@article{anwar2019ridnet,
  title={Real Image Denoising with Feature Attention},
  author={Anwar, Saeed and Barnes, Nick},
  journal={IEEE International Conference on Computer Vision (ICCV-Oral)},
  year={2019}
}
```
## Acknowledgements
This code is built on [DRLN (PyTorch)](https://github.com/saeed-anwar/DRLN)

