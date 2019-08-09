import random
import numpy as np
import matplotlib.pyplot as plt 
import os
import collections
import pandas as pd

from torchvision import transforms, datasets
import torch

import models

def scale(x, features_range=(-1,1)):
    """
        Recale takes in an image x and returns that image, scaled
        with a feature_range of pixel values in features_range. 
    """

    #Rescale x between 0. and 1.
    minix = x.min() 
    maxix = x.max()
    x = (x-minix)/maxix

    #Rescaling to features_range
    mini = np.min(features_range)
    maxi = np.max(features_range)
    length = maxi - mini
    center = length/2.

    return length*x - center

def viz_data_img(images,n=0,labels=None):
    """
        Plot tensor in images, in greyscale
    """
    fig = plt.figure(n,figsize=(25, 4))
    plot_size=images.shape[0]
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze(),cmap = 'Greys_r')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        if labels != None:
            ax.set_title(str(labels[idx].item()))


def recup_tested_hyperparameters(folder = '.'):
    """
        Recuperate the hyperparameters already tested
    """
    hyperparameters_tested = []

    if os.path.exists(folder + '/Tested_hyperparameters.csv'):
        with open(folder + '/Tested_hyperparameters.csv','r') as fichier:
            content = fichier.read()

        content = content.split('\n')
        
        if content[-1] == '':
            del content[-1]

        for line in content:
            if line != '':
                line = line.split(';')
                parameters = []
                for parameter in line:
                    if '.' in parameter or 'e' in parameter:
                        parameters.append(float(parameter))
                    else:
                        parameters.append(int(parameter))

                hyperparameters_tested.append(parameters)

    return hyperparameters_tested


def select_hyperparameters(file_hyperparameters, separator = ';'):
    """
        Select randomly hyperparameters in file file_hyperparameters
        Verify if the hyperparamaters are already tested
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    keys = content[0].split(separator)
    del content[0]
    del content[-1]

    hyperparameters_possible = []

    for i in range(len(keys)):
        hyperparameters_possible.append([])

    for line in content:
        line = line.split(separator)
        for i in range(len(line)):
            if line[i] != '':
                if ',' in line[i]:
                    hyperparameters_possible[i].append(float(line[i].replace(',','.')))
                else:
                    hyperparameters_possible[i].append(int(line[i]))

    nb_combinaison_possible = 1

    for parameter in hyperparameters_possible:
        nb_combinaison_possible *= len(parameter)

    hyperparameters_tested = recup_tested_hyperparameters()

    hyperparameters = []

    tested = True

    while tested and len(hyperparameters_tested) <= nb_combinaison_possible :
        for parameter in hyperparameters_possible:
            hyperparameters.append(parameter[random.randint(0,len(parameter)-1)])
        tested = hyperparameters in hyperparameters_tested    
        
    return collections.OrderedDict(zip(keys, hyperparameters)) 


def select_hyperparameters_row(file_hyperparameters, separator = ';'):
    """
        Select randomly hyperparameters in file file_hyperparameters
        Verify if the hyperparamaters are already tested
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    keys = content[0].split(separator)
    
    del content[0]
    
    if len(content[-1]) == 0:    
        del content[-1]

    hyperparameters_possible = []

    for line in content:
        line = line.split(separator)
        hyperparameters = []
        
        for i in range(len(line)):
            if line[i] != '':
                if ',' in line[i] or 'e' in line :
                    hyperparameters.append(float(line[i].replace(',','.')))
                else:
                    hyperparameters.append(int(line[i]))
        hyperparameters_possible.append(hyperparameters)

    nb_hyperparameters_possible = len(content)

    hyperparameters_tested = recup_tested_hyperparameters()

    hyperparameters = []

    tested = True

    while tested and len(hyperparameters_tested) <= nb_hyperparameters_possible :
        hyperparameters = hyperparameters_possible[random.randint(0, len(hyperparameters_possible)-1)]
        tested = hyperparameters in hyperparameters_tested    

    return collections.OrderedDict(zip(keys, hyperparameters)) 


def recup_hyperparameters(file_hyperparameters, separator = ' = '):
    """
        Recuperate the hyperparameters of a NN
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    hyperparameters = []
    keys = []

    for line in content:
        line = line.split(separator)
        keys.append(line[0])
        if '.' in line[1] or 'e' in line[1]:
            hyperparameters.append(float(line[1]))
        else:
            hyperparameters.append(int(line[1]))

    return collections.OrderedDict(zip(keys, hyperparameters))

class EncodedImages(torch.utils.data.Dataset):
    """Encoded Imagess dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.encoded_images = pd.read_csv(csv_file, header = None, sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.encoded_images)

    def __getitem__(self, idx):
        image = self.encoded_images.iloc[idx,:-1]
        #image = image.astype('float')
        image = np.ndarray(shape=(len(image),), buffer = np.array(image))
        label = self.encoded_images.iloc[idx, -1]
        #label = label.astype('int')
        label = np.ndarray(shape=(1,), buffer = np.array([label]),dtype='int')
        sample = {'image' : image, 'label' : label}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


def encode_images(data_path, hyperparameters, nb_batch = 100):

    folder = '../Encoded_images/AE'

    for parameter in hyperparameters.values():
        folder += '_' + str(parameter) 

    if not os.path.exists(folder):
        os.makedirs(folder)

    if len(os.listdir(folder)) == 0:

        print('On encode les images')

        os.chdir(folder)
        
        latent_size = hyperparameters['latent_size']

        transform = transforms.Compose([transforms.Resize(80),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor()
                                        ])

        train_data = datasets.ImageFolder('../' + data_path, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data)//nb_batch, shuffle=False)

        Encoder = models.Critic(64,64, latent_size = latent_size, mode='AE', nb_channels=3)
        state_dict = torch.load('../../../checkpoints/encoder.pth')
        Encoder.load_state_dict(state_dict)

        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            Encoder.cuda()
        
        filename = 'encoded_images.csv'
        
        for images, labels in train_loader:
            if train_on_gpu:
                images = images.cuda()

            images = scale(images)

            with torch.no_grad():
                encoded_images = Encoder.forward(images)
                
                encoded_images = encoded_images.cpu()

                for i in range(encoded_images.shape[0]):
                    with open(filename, 'a') as fichier:
                        image = encoded_images[i].numpy()
                        for j in range(image.shape[0]):
                            fichier.write(str(image[j]))
                            fichier.write(';')
                        fichier.write(str(int(labels[i])))
                        fichier.write('\n')
        os.chdir('..')

    return folder + '/encoded_images.csv'

def recup_scores(score, folder, bigger_is_better=False, nb_values = 10):

    vect = np.ones(nb_values)/nb_values
    
    hyperparamaters = recup_hyperparameters(folder + '/Hyperparameters.txt')
    epochs = hyperparamaters['epochs']

    if not os.path.exists(folder + '/log.csv'):
        if bigger_is_better:
            return 0
        else:
            return np.inf
    else:
        datas = pd.read_csv(folder + '/log.csv', sep=';')
        if len(datas) == epochs:
            Score_mean = np.dot(datas[score].iloc[-nb_values:].values, vect)
        else:
            if bigger_is_better:
                return 0
            else:
                return np.inf

    return Score_mean
    