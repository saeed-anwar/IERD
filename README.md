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
  <img width="500" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Main.png">
</p>

Denoising results: In the first row, an image is corrupted by the Gaussian noise with σ = 50 from the BSD68 dataset. In the second row, a sample image from the RNI15 real noisy dataset. Our results have the best PSNR score for synthetic images, and unlike other methods, it does not have over-smoothing or over-contrasting artifacts. 

## Network
The proposed network architecture, which consists of multiple modules with similar structures. Each module is composed of a series of pre-activation-convolution layer pairs. The multiplier block negates the input block features to be summed at the end of the mapping module

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/Network.png">
</p>


## Requirements
- The model is built in PyTorch 0.4.0, PyTorch 0.4.1 
- Tested on Ubuntu 14.04/16.04 environment 
- python 3.6
- CUDA 9.0 
- cuDNN 5.1 
- pytorch=0.4.1
- torchvision=0.2.1
- imageio
- pillow

## Test
### Quick start
1. Download the trained models and code of our paper from [Google Drive](https://drive.google.com/file/d/1DV9-OgvYoR4ELQZY-R7vZiX5nTf-NZtX/view?usp=sharing). The total size for all models is **3.1MB.**

2. cd to '/IERDTestCode/code', run the following scripts and find the results in directory **IERD_Results**.

    **You can use the following script to test the algorithm. The first script is without self-ensembler and the second one is with self-ensemble.**

``` #Normal
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model IERD --n_feats 64 --pre_train ../trained_model/IERD.pt --test_only --save_results --save 'SSID_Results' --testpath ../noisy --testset SIDD
```

``` #Ensemble
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model IERD --n_feats 64 --pre_train ../trained_model/IERD.pt --test_only --save_results --save 'SSIDPlus_Results' --testpath ../noisy --testset SIDD --self_ensemble
```


## Results
**All the results for IERD can be downloaded from GoogleDrive from**  [SSID](https://drive.google.com/file/d/1em70fbrVCggxdv1vi0dLqriAR_f2lPjc/view?usp=sharing) (118MB), [RNI15](https://drive.google.com/file/d/1NUmFpS7Zl4f70OZJVd96t35wSGSyvfMS/view?usp=sharing) (9MB) and [DnD](https://drive.google.com/file/d/1IfTi6ZImNsrzqC6oFhgFF8Z9QKvZeAfE/view?usp=sharing) (2.3GB). 

### DnD Results

Comparison of our method against the state-of-the-art algorithms on real images containing Gaussian noise from Darmstadt Noise Dataset (DND) benchmark for different denoising algorithms. Difference can be better viewed in magnified view.
<p align="center">
  <img width="800" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/DnDFig.png">
</p>

Mean PSNR and SSIM of the denoising methods evaluated on the real images dataset
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/DnDTable.png">
</p>


### SSID Results

A few challenging examples from SSID dataset. Our method can restore true colors and remove noise.
<p align="center">
  <img width="800" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/SSID.png">
</p>

The quantitative results (in PSNR (dB)) for the SSID dataset
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/SSIDTable.png">
</p>

### RNI15

Sample visual examples from RNI15. Our method annihilates the noise and preserves the essential details while the competing methods fail to deliver satisfactory results i.e. unable to remove noise. Best viewed on high-resolution display.

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/MoreResults.png">
</p>

## Ablation Studies

### Effect of Patch Size

<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/trainingpatchsize.png">
</p>


### Layers vs. Dilation

<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/layersvsdilation.png">
</p>


### Effect of Number of Modules

<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/NumberofModules.png">
</p>


### Effect of Each Component

<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/CompoenentsComparison.png">
</p>


### Structure of Identity Module

<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/identityModules.png">
</p>

For more information, please refer to our [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Anwar_Identity_Enhanced_Residual_Image_Denoising_CVPRW_2020_paper.pdf)

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

