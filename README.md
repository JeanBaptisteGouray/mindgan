<h1 align="center">
  <br>
  Mind2Mind : transfer learning for GANs 
  <br>
</h1>

  <p align="center">
    <a >Jean-Baptiste Gouray </a> •
    <a >Yaël Frégier</a> 

<h4 align="center">Official repository of the paper</h4>


<p align="center">
  <img src="./random_choice.png">  </p>

This repository contains a Mind2Mind transfer module. We have added it to a fork of <a href="https://github.com/podgorskiy/ALAE"> the ALAE repository</a>. We have kept from this fork only the modules essential for running Mind2Mind. If you need the full capacities of ALAE, add our module to the original ALAE repository.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/pytorch-1.4.0-green.svg?style=plastic" alt="pytorch version">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

  <p align="center">
    <a href="https://drive.google.com/file/d/17ZUtqR6nYw1pUxzKKyb5EJUTZF7s1SYW/view?usp=sharing">Google Drive folder with models and qualitative results</a>
  </p>


# Mind2Mind

> **Transfer Learning for GANs**
>
> **Abstract:** *Training generative adversarial networks (GANs) on high quality (HQ) images involves important computing resources. This requirement represents a bottleneck for the development of applications of GANs. We propose a transfer learning technique for GANs that significantly reduces training time. Our approach consists of freezing the low-level layers of both the critic and generator of the original GAN. We assume an auto-encoder constraint in order to ensure the compatibility of the internal representations of the critic and the generator. This assumption explains the gain in training time as it enables us to bypass the low-level layers during the forward and backward passes. We compare our method to baselines and observe a significant acceleration of the training. It can reach two orders of magnitude on HQ datasets when compared with StyleGAN. We prove rigorously, within the framework of optimal transport, a theorem ensuring the convergence of the learning of the transferred GAN. We moreover provide a precise bound for the convergence of the training in terms of the distance between the source and target dataset.*



## Repository organization

To run the scripts, you will need to have a CUDA capable GPU, PyTorch >= v1.3.1 and cuda/cuDNN drivers installed.
Install the required packages:

    pip install -r requirements.txt
  


#### Running scripts

The code in the repository is organized in such a way that all scripts must be run from the root of the repository.
If you use an IDE (e.g. PyCharm or Visual Studio Code), just set *Working Directory* to point to the root of the repository.

If you want to run from the command line, then you also need to set **PYTHONPATH** variable to point to the root of the repository.

For example, let's say we've cloned repository to *~/ALAE* directory, then do:

    $ cd ~/ALAE
    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

![pythonpath](https://podgorskiy.com/static/pythonpath.svg)

Now you can run scripts as follows:

    $ python module_mind/generate_images.py

#### Repository structure


| Path | Description
| :--- | :----------
| ALAE | Repository root folder
| &boxvr;&nbsp; configs | Folder with yaml config files.
| &boxv;&nbsp; &boxur;&nbsp; ffhq.yaml | Config file for FFHQ dataset at 1024x1024 resolution.
| &boxvr;&nbsp; module_mind | Folder with Mind2Mind module.
| &boxv;&nbsp; &boxvr;&nbsp; data_loader.py | Class to define loaders for encoded data.
| &boxv;&nbsp; &boxvr;&nbsp; download_mindGAN.py | Script to download a pre-trained MindGan.
| &boxv;&nbsp; &boxvr;&nbsp; generate_images.py | Script to generate images from the MindGAN. 
| &boxv;&nbsp; &boxvr;&nbsp; model.py | MindGAN model.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_data.py | Script to download celebaHQ and encode the data. 
| &boxv;&nbsp; &boxvr;&nbsp; train.py | Script to train the MindGAN on CelebaHQ from the ALAE autoencoder trained FFHQ.
| &boxv;&nbsp; &boxur;&nbsp; trainer.py | Class for handling training loops.
| &boxvr;&nbsp; checkpointer.py | Module for saving/restoring model weights, optimizer state and loss history.
| &boxvr;&nbsp; defaults.py | Definition for config variables with default values.
| &boxvr;&nbsp;  losses.py | Defintions of the loss fonctions.
| &boxvr;&nbsp;  lreq.py  |  Custom `Linear`, `Conv2d` and `ConvTranspose2d` modules for learning rate equalization.
| &boxvr;&nbsp; model.py | Module with high-level model definition.
| &boxvr;&nbsp;  net.py | Definition of all network blocks for multiple architectures.
| &boxvr;&nbsp;  registry.py | Registry of network blocks for selecting from config file.
| &boxvr;&nbsp; random_choice.png | Sample of images (for this readme).
| &boxvr;&nbsp;  requirements.txt | List of python modules needed.
| &boxur;&nbsp; utils.py | Decorator for async call, decorator for caching, registry for network blocks.


#### Configs



In ALAE, you can specify which **yaml** config [**yacs**](https://github.com/rbgirshick/yacs)  will use. However, our Mind2Mind module only accepts for the moment the `ffhq` config. Since it is the default config for ALAE you do not have anything to do. However, if you know the use of the `-c` parameter from ALAE, do not try to use it here to choose another config.

#### Datasets

You must prepare the data with the command:

    $ python module_mind/prepare_data.py
    

#### Pre-trained models

To download pre-trained models run:

    python module_mind/download_mindGAN.py



## Generating figures

To make generation figure run:

    python module_mind/generate_images.py 
    
By default, it will generate one batch of 4 images. If you want to modify the numbers of batches and images, you have to modify the lines 19-20 in `generate_images`. In particular if your system runs out of memory, you will need to lower the number of images par batch and restart the kernel. The generated samples can be found in the folder `module_mind/images_generated/mind2mid`.

## Training

To run training:

    python module_mind/train.py 
    
We have only tested our Mind2Mind module on a single GPU.

You might need to adjust the batch size in the config file depending on the memory size of the GPU.

## Computation of FID

To compute the fid score, you need to download the module <a href="https://github.com/mseitzer/pytorch-fid"> pytorch-fid</a> and run the command :
    
    python fid_score $ALAE_PATH/module_mind/Dataset/Celeba-HQ/data1024x1024 $ALAE_PATH/module_mind/images_generated/mind2mind/
    
where `$ALAE_PATH` is the directory in which ALAE is located.