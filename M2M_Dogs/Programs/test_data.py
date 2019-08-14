import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models as pretrain
import torch.optim as optim

import utils
import models
import save
import helper

import os
import numpy as np
import time
import shutil

data_path = '/media/Data/Test'


folder = '../Classifier'

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + '/Hyperparameters.csv'):
    shutil.copyfile('../../Hyperparameters/Hyperparameters_classifier.csv',  folder + '/Hyperparameters.csv')

os.chdir(folder)

hyperparameters = utils.select_hyperparameters('Hyperparameters.csv')

# Hyperparameters for the training
[batch_size, num_workers, conv_dim, lr, beta1, beta2, epochs] = list(hyperparameters.values())

# Transformation and datasets
transform = transforms.Compose([transforms.Resize(140),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

data_path, train_loader, test_loader = utils.dataset(data_path, 'Dog_Breed', batch_size, transform)